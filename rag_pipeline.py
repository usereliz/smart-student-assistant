"""
rag_pipeline.py
---------------
Core RAG logic:
  1. Load PDFs and split into chunks
  2. Embed chunks and store in ChromaDB
  3. Retrieve relevant chunks for a query
  4. Generate an answer using Gemini
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_db"


# ── 1. Load & Split ────────────────────────────────────────────────────────────

def load_and_split_pdfs(pdf_paths: list[str]) -> list:
    """Load PDF files and split them into overlapping chunks."""
    all_docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # characters per chunk
        chunk_overlap=100,     # overlap keeps context across chunks
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(all_docs)
    return chunks


# ── 2. Embed & Store ───────────────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and store them in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    # Split into small batches to avoid timeout
    import time
    batch_size = 10
    all_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    
    vectorstore = None
    for i, batch in enumerate(all_batches):
        print(f"Indexing batch {i+1}/{len(all_batches)}...")
        for attempt in range(3):  # retry up to 3 times
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=CHROMA_PERSIST_DIR,
                    )
                else:
                    vectorstore.add_documents(batch)
                time.sleep(1)  # small pause between batches
                break
            except Exception as e:
                print(f"Retry {attempt+1} after error: {e}")
                time.sleep(3)
    
    vectorstore.persist()
    return vectorstore

def load_vectorstore() -> Chroma | None:
    """Load an existing ChromaDB vectorstore from disk."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


# ── 3. Build QA Chain ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful study assistant for a computer science student.
Answer questions based ONLY on the provided course documents.
If the answer is not in the documents, say: "I couldn't find that in your course materials."

Always be:
- Clear and concise
- Student-friendly (explain jargon if needed)
- Honest when you don't know something

Context from your documents:
{context}

Chat history:
{chat_history}

Student question: {question}
Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)


def build_qa_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Build a conversational RAG chain with memory."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # free tier model
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,            # lower = more factual answers
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.4},
)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return chain


# ── 4. Ask a Question ──────────────────────────────────────────────────────────

def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    result = chain({"question": question})

    source_docs = result.get("source_documents", [])

    # Confidence filter: if no docs retrieved, don't hallucinate
    if not source_docs:
        return {
            "answer": "I couldn't find relevant information in your course materials for this question.",
            "sources": [],
            "low_confidence": True,
        }

    # Extract unique sources
    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        file_name = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page", 0) + 1
        key = (file_name, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": file_name, "page": page})

    return {
        "answer": result["answer"],
        "sources": sources,
        "low_confidence": False,
    }
