"""
app.py
------
Streamlit UI for the Smart Student Assistant.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import (
    load_and_split_pdfs,
    build_vectorstore,
    load_vectorstore,
    build_qa_chain,
    ask_question,
)

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Student Assistant",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { max-width: 780px; }
    .source-badge {
        display: inline-block;
        background: #f0f4ff;
        color: #3b5bdb;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px 4px 2px 0;
        border: 1px solid #dbe4ff;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── Sidebar: Upload PDFs ───────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Your Documents")
    st.caption("Upload your course PDFs to get started.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Reading and indexing your PDFs... this takes ~30 seconds"):
                # Save uploaded files to a temp directory
                tmp_paths = []
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for f in uploaded_files:
                        tmp_path = os.path.join(tmp_dir, f.name)
                        with open(tmp_path, "wb") as out:
                            out.write(f.read())
                        tmp_paths.append(tmp_path)

                    # Build the RAG pipeline
                    chunks = load_and_split_pdfs(tmp_paths)
                    vectorstore = build_vectorstore(chunks)

                st.session_state.qa_chain = build_qa_chain(vectorstore)
                st.session_state.docs_loaded = True
                st.session_state.messages = []  # reset chat on new upload

            st.success(f"✅ Indexed {len(uploaded_files)} file(s)!")

    # Try loading existing vectorstore
    if not st.session_state.docs_loaded:
        existing = load_vectorstore()
        if existing:
            st.session_state.qa_chain = build_qa_chain(existing)
            st.session_state.docs_loaded = True
            st.info("📂 Loaded previously indexed documents.")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Upload your course PDFs
2. Click **Index Documents**
3. Ask any question about your course
4. Get answers with source references
    """)

    if st.session_state.docs_loaded:
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.session_state.qa_chain = None
            st.session_state.docs_loaded = False
            st.session_state.messages = []
            st.rerun()

# ── Main Chat Area ─────────────────────────────────────────────────────────────

st.title("🎓 Smart Student Assistant")

if not st.session_state.docs_loaded:
    st.info("👈 Upload your course PDFs in the sidebar to get started.")

    # Example questions to inspire the user
    st.markdown("#### What can you ask?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- 📖 *Explain what a linked list is*
- 🔍 *What is the time complexity of quicksort?*
- 💡 *Summarize chapter 3*
        """)
    with col2:
        st.markdown("""
- 🧠 *What is the difference between TCP and UDP?*
- 📝 *List the key points about SQL joins*
- ❓ *What did the professor say about recursion?*
        """)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                source_html = "".join(
                    f'<span class="source-badge">📄 {s["file"]} · p.{s["page"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your course..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                result = ask_question(st.session_state.qa_chain, prompt)

            st.markdown(result["answer"])

            if result["sources"]:
                source_html = "".join(
                    f'<span class="source-badge">📄 {s["file"]} · p.{s["page"]}</span>'
                    for s in result["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
