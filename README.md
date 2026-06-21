# RAG Application - Chat with Your PDFs

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about your PDF documents.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/em-sharjeel22/RAG_APPLICATION.git
cd RAG_APPLICATION
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your PDFs
- Create or use the `data/` folder
- Place all PDF files in `data/`

### 5. Build the Search Index (first time only)
```bash
python main.py --ingest
```

### 6. Start Chatting
```bash
python main.py --chat
```

Or ask a single question:
```bash
python main.py --question "What is this about?"
```

## Project Structure
```
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── data/               # Your PDF files here
├── faiss_index/        # Search index (auto-created)
└── src/
    ├── config.py       # Configuration settings
    ├── connections.py  # LLM & embedding setup
    ├── ingest.py       # PDF ingestion
    ├── retriever.py    # RAG logic
    └── vector_store.py # Vector database
```

## Requirements
- Python 3.11+
- GROQ API Key (free at https://console.groq.com)

## Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running on Streamlit

### Local Development
```bash
# Terminal 1: Start the API
uvicorn api:app --reload --port 8000

# Terminal 2: Run Streamlit
streamlit run streamlit_ui.py
```

### Streamlit Cloud Deployment
1. Push this repo to GitHub
2. Create account at https://streamlit.io
3. Click "Create app" and connect your GitHub repo
4. Set secrets in Streamlit Cloud:
   - Go to Settings → Secrets
   - Add: `GROQ_API_KEY = your_key_here`
   - Add: `API_URL = https://your-api-endpoint.com` (if using external API)
5. Deploy!
