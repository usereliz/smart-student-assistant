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

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/smart-student-assistant
cd smart-student-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your free Gemini API key
- Go to https://aistudio.google.com
- Click "Get API key" → Create API key
- Copy the key

### 3. Set up environment variables
```bash
cp .env.example .env
# Open .env and paste your API key
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Deploy free on Streamlit Cloud
- Push your code to GitHub
- Go to https://share.streamlit.io
- Connect your repo → Deploy
- Add your GEMINI_API_KEY in the Secrets section

## 🛠️ How it works

1. **Upload** your course PDFs
2. **Indexing**: PDFs are split into chunks → converted to embeddings → stored in ChromaDB
3. **Querying**: Your question is embedded → similar chunks are retrieved → Gemini generates an answer using those chunks as context
4. **Answer**: Response shown with the source document and page number

## 📌 CV Description
> Smart Student Assistant — Python, LangChain, ChromaDB, Gemini API, Streamlit  
> Built a RAG-based chatbot that answers questions from uploaded course PDFs. Deployed on Streamlit Cloud.
