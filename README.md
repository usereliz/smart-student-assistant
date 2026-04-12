# 🎓 Assistant Étudiant Intelligent

Un chatbot basé sur la technique RAG qui répond à tes questions à partir de tes cours en PDF grâce à l'IA.
Technologies utilisées : Python · LangChain · ChromaDB · Google Gemini · Streamlit

📁 Structure du projet
smart-student-assistant/
├── app.py                  # Application Streamlit principale
├── rag_pipeline.py         # Logique RAG (chargement, embeddings, recherche, génération)
├── requirements.txt        # Toutes les dépendances
├── .env.example            # Modèle pour les variables d'environnement
└── README.md

## 🚀 Installation & Lancement

1. Cloner & installer
```bash
git clone https://github.com/usereliz/smart-student-assistant
cd smart-student-assistant
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
2. Lancer l'application
```bash
streamlit run app.py
```
## 🛠️ Comment ça fonctionne

1. **Import** de tes cours en PDF
2. **Indexation** : les PDFs sont découpés en morceaux → convertis en embeddings → stockés dans ChromaDB
3. **Recherche** : ta question est convertie en embedding → les passages les plus similaires sont récupérés → Gemini génère une réponse basée sur ces passages
4. **Réponse** : la réponse s'affiche avec le document source et le numéro de page
