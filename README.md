# Free RAG AI Chatbot

This project is a simple Retrieval-Augmented Generation (RAG) chatbot built with free and open-source tools.

It uses:
- Python + FastAPI for the backend
- Sentence-transformers for embeddings
- FAISS for vector search
- Ollama for free local LLM inference
- A lightweight HTML/JS frontend

Features:
- Chat with your own documents
- Local, no paid API required
- PDF/TXT/MD knowledge base support
- Simple web interface
- Easy setup for local development

## Tech stack
- Backend: FastAPI
- Embeddings: sentence-transformers
- Vector database: FAISS
- LLM: Ollama (free local models)
- Frontend: HTML + JavaScript

## Project structure

```text
rag-ai-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── rag.py
│   ├── main.py
│   └── models.py
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── data/
│   └── knowledge/
│       └── sample.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
└── run.py
```

## Quick start (local)

### 1) Install Ollama

Download and install Ollama from: https://ollama.com/download

Then pull a free model:

```bash
ollama pull llama3.2
```

You can also use other free models such as:
- llama3.2
- mistral
- qwen2.5
- phi3

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start the app

```bash
python run.py
```

Then open:

```text
http://localhost:8000
```

## Docker setup

This project also includes Docker support.

```bash
docker compose up --build
```

This starts:
- Ollama service on port 11434
- FastAPI app on port 8000

## Adding your own knowledge

Put your documents in:

```text
data/knowledge/
```

Supported file types:
- .txt
- .md
- .pdf

The app will automatically ingest them when the server starts.

## How it works

1. Your documents are split into chunks.
2. Each chunk is embedded using a free sentence-transformer model.
3. The chunks are stored in a FAISS vector index.
4. When you ask a question, the system retrieves the most relevant chunks.
5. Those chunks are sent to the local Ollama model as context.
6. The model answers based on your knowledge base.

## Configuration

Copy the example file:

```bash
cp .env.example .env
```

Edit the values as needed.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Notes

- This project intentionally uses local free tools only.
- It works best with a machine that can run a smaller local LLM.
- If Ollama is not running, the app will show a clear error message.

## License

MIT

## Author

Built for local AI experimentation and learning.
