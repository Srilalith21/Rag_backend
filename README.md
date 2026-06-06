# YouTube RAG Chatbot — Django Backend

A pure-Django REST API that lets users ask natural-language questions about any YouTube video.

## How it works

```
YouTube URL → Transcript → Chunks → Embeddings → FAISS index
                                                        ↓
User question → Query embedding → Similarity search → LLM → Answer
```

## Tech stack

| Layer | Library |
|---|---|
| Web framework | Django + Django REST Framework |
| Transcript | youtube-transcript-api |
| Embeddings & LLM | OpenAI (text-embedding-ada-002, gpt-3.5-turbo) |
| Vector store | FAISS (persisted to disk) |
| RAG orchestration | LangChain (LCEL) |

---

## Quick start

### 1. Clone & install

```bash
git clone <repo>
cd youtube_rag_backend
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
python manage.py migrate   # creates sqlite db (not really needed, but good practice)
python manage.py runserver
```

Server is live at `http://127.0.0.1:8000`

---

## API Endpoints

### POST `/api/load-video/`
Load a YouTube video — extracts transcript and builds FAISS index.

**Request:**
```json
{ "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }
```

**Response:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "message": "Video loaded successfully. Ready to chat!",
  "chunk_count": 42,
  "already_indexed": false
}
```

---

### POST `/api/chat/`
Ask a question about a loaded video.

**Request:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "question": "What is the main topic of this video?"
}
```

**Response:**
```json
{
  "answer": "The video is about ...",
  "source_chunks": ["...transcript chunk 1...", "...transcript chunk 2..."]
}
```

---

### GET `/api/video-status/?video_id=dQw4w9WgXcQ`
Check if a video is already indexed (avoids re-processing).

**Response:**
```json
{ "video_id": "dQw4w9WgXcQ", "indexed": true }
```

---

## Project structure

```
youtube_rag_backend/
├── manage.py
├── requirements.txt
├── .env.example
├── faiss_indexes/          ← auto-created; one folder per video_id
├── youtube_rag/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── chatbot/
    ├── utils.py            ← transcript fetch, FAISS build/load, RAG chain
    ├── views.py            ← DRF APIViews
    └── urls.py
```

## Settings you can tune (settings.py)

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per transcript chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVER_K` | 4 | Top-k chunks retrieved per query |
| `FAISS_INDEX_DIR` | `<BASE_DIR>/faiss_indexes` | Where indexes are stored |
