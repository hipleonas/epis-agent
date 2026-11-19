"# 🎓 Teacher Chatbot - Trợ Lý Học Tập Thông Minh

LangGraph + LangChain chatbot with RAG capabilities using Zilliz (Managed Milvus) vector database and OpenAI.

## 🌟 Features

- **Intelligent Intent Routing**: Automatically detects if user wants normal chat or learning content
- **RAG-Based Learning**: Retrieves relevant content from uploaded course materials
- **Multi-Course Support**: Filter by course and module for precise content retrieval
- **Vietnamese Language**: Optimized for Vietnamese educational content
- **Teacher Persona**: Responds like a helpful, encouraging teacher

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Zilliz Cloud account (free tier available at https://cloud.zilliz.com/)

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Zilliz Cloud

1. Go to https://cloud.zilliz.com/ and sign up
2. Create a new cluster (free tier is fine)
3. Get your cluster endpoint (URI) and API key
4. Update `.env` with your credentials

**Zilliz Configuration:**
- **Metric Type**: Cosine
- **Index Type**: IVF_FLAT (auto-created)
- **Dimension**: 1536 (text-embedding-3-small)
- **Collection**: learning_content

### 3. Ingest Sample Data

```powershell
python scripts/ingest_data.py
```

### 4. Run Chatbot

```powershell
python app.py
```

## 📊 Vector Database Schema

**Collection**: `learning_content`

| Field      | Type         | Description                    |
|------------|--------------|--------------------------------|
| id         | VARCHAR(512) | Unique identifier              |
| embedding  | FLOAT_VECTOR | 1536-dim embedding             |
| text       | VARCHAR      | Original text content          |
| course_id  | VARCHAR(256) | Course ID (can be empty)       |
| module_id  | VARCHAR(256) | Module ID                      |

**Metadata filtering**: Search is filtered by `course_id` and/or `module_id`

## 📁 Project Structure

```
gsai-chatbot/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
├── src/
│   ├── embeddings.py           # OpenAI embeddings
│   ├── vectorstore.py          # Zilliz operations
│   ├── rag.py                  # RAG module
│   ├── agent.py                # LangGraph agent
│   └── utils.py                # Text chunking
├── scripts/
│   └── ingest_data.py          # Data ingestion
└── docs/
    └── ZILLIZ_SETUP.md         # Detailed setup guide
```

## 🌐 API Deployment

Deploy as a REST API for web/mobile integration:

```powershell
# Test locally
python api.py

# Visit http://localhost:8000/docs for interactive API docs
```

**API Endpoints:**
- `POST /chat` - Chat with AI (with RAG)
- `POST /upload/pdf` - Upload and embed PDF files
- `GET /health` - Health check

**Deploy to Render:**
1. Push code to GitHub
2. Connect repository to Render
3. Add environment variables
4. Deploy!

See [API_GUIDE.md](docs/API_GUIDE.md) for complete deployment guide and [PDF_UPLOAD.md](docs/PDF_UPLOAD.md) for PDF upload documentation.

See [ZILLIZ_SETUP.md](docs/ZILLIZ_SETUP.md) for detailed configuration guide" 
