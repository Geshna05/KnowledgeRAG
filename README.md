# KnowledgeRAG

Upload PDF notes and/or YouTube videos, chat with the content, get answers
with citations (PDF page number / video timestamp), and auto-generate quizzes.

## Architecture

```
PDF -----> pypdf (per-page text) --------\
                                           >--- chunk + embed (local) ---> ChromaDB (persistent)
YouTube -> youtube-transcript-api ------/                                       |
                                                                                 v
User question --> retrieve top-k relevant chunks --> local LLM via Ollama --> answer + citations
                                                                                 |
                                                                                 v
                                                                      SQLite (chat history)

Quiz: pull chunk text for chosen doc(s) --> local LLM via Ollama --> structured JSON quiz
```

## Setup

### 1. Install Ollama (the free local LLM engine)

Go to https://ollama.com/download and install it for your OS (macOS, Windows, or Linux). This gives you an `ollama` command and a local server.

### 2. Pull a model

```bash
ollama pull llama3.1
```

This downloads the model once (a few GB) and it then runs fully offline. If your machine is low on RAM/VRAM, use a smaller model instead:

```bash
ollama pull phi3          # ~2.3GB, runs fine on most laptops
# or
ollama pull llama3.2:3b   # small and fast
```

If you use a different model name than `llama3.1`, set `OLLAMA_MODEL` in `.env` to match (see step 4).

Ollama usually starts its server automatically after install. If not, run:
```bash
ollama serve
```
Leave that running in the background (or just keep the Ollama desktop app open).

### 3. Python environment

```bash
cd study-assistant-rag
python3 -m venv venv
Windows: venv\Scripts\activate #source venv/bin/activate        
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
```

Nothing to fill in unless you picked a different model — if so, edit `.env`:
```
OLLAMA_MODEL=phi3
```

## Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Creates `chroma_data/` (vector DB) and `study_assistant.db` (chat history) on first run.

## Run the frontend

Single static HTML file, no build step:

```bash
open frontend/index.html                 # macOS
xdg-open frontend/index.html             # Linux
Windows: double-click frontend/index.html
```

Talks to the backend at `http://localhost:8000` (edit `API_BASE` at the top of the `<script>` in `index.html` if needed).

## Use it
1. Upload a PDF or paste a YouTube URL in the sidebar.
2. Wait for it to appear under "Your Materials."
3. Ask questions in the Chat tab — answers include citation chips (page number / timestamp deep-link).
4. Generate a quiz from the Quiz tab.

## API summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/documents/upload-pdf` | POST (multipart) | Upload + index a PDF |
| `/documents/upload-youtube` | POST `{url, title?}` | Fetch transcript + index a video |
| `/documents` | GET | List uploaded documents |
| `/documents/{id}` | DELETE | Remove a document and its chunks |
| `/sessions` | POST | Start a new chat session |
| `/sessions/{id}/history` | GET | Get chat history for a session |
| `/chat` | POST `{session_id, message, document_ids?}` | Ask a question, get answer + citations |
| `/quiz` | POST `{document_ids?, num_questions?, difficulty?}` | Generate a quiz |

`document_ids` is optional — omit it (or leave all checkboxes ticked) to search/quiz across everything uploaded.


### Chat

![Chat](screenshots/CHAT.png)

### MCQ Quiz - Example 1

![MCQ Quiz 1](screenshots/MCQ1.png)

### MCQ Quiz - Example 2

![MCQ Quiz 2](screenshots/MCQ2.png)


