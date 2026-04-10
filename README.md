# 🎓 Smart Student Assistant

A RAG-based chatbot that answers questions from your course PDFs using AI.
Built with: Python · LangChain · ChromaDB · Google Gemini · Streamlit

## 📁 Project Structure

```
smart-student-assistant/
├── app.py                  # Main Streamlit app
├── rag_pipeline.py         # RAG logic (load, embed, retrieve, generate)
├── requirements.txt        # All dependencies
├── .env.example            # Environment variables template
└── README.md
```

## 🚀 Setup & Run

1. Clone & install
```bash
git clone https://github.com/usereliz/smart-student-assistant
cd smart-student-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the app
```bash
streamlit run app.py
```

## 🛠️ How it works

1. Upload your course PDFs
2. Indexing: PDFs are split into chunks → converted to embeddings → stored in ChromaDB
3. Querying: Your question is embedded → similar chunks are retrieved → Gemini generates an answer using those chunks as context
4. Answer: Response shown with the source document and page number
