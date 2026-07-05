"""
Retrieval + chat: pulls the most relevant chunks for a question, feeds them
to a local LLM (via Ollama) as grounding context, and returns an answer plus
a citations list (page number for PDFs, timestamp + deep-link for YouTube).

Uses Ollama (https://ollama.com) running locally - free, no API key,
no internet required after the model is pulled once.
"""
import os
from typing import List, Optional

import requests

from . import vectorstore, db

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

SYSTEM_PROMPT = """You are a focused study assistant.
Answer the user's question using ONLY the provided context excerpts from their
uploaded notes/videos. Reference which excerpt(s) you used inline using
bracket numbers, e.g. [1], [2], matching the excerpt numbers given to you.
If the answer is not contained in the context, say so plainly rather than
guessing. Keep answers clear and exam-prep friendly."""


def format_context(hits: List[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        if meta["source_type"] == "pdf":
            loc = f"page {meta['page']}"
        else:
            loc = f"timestamp {int(meta['start_time'])}s"
        lines.append(f"[{i}] (Source: {meta['source_name']}, {loc})\n{h['text']}")
    return "\n\n".join(lines)


def build_citations(hits: List[dict]) -> List[dict]:
    citations = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        entry = {
            "ref": i,
            "source_name": meta["source_name"],
            "source_type": meta["source_type"],
        }
        if meta["source_type"] == "pdf":
            entry["page"] = meta["page"]
        else:
            start = int(meta["start_time"])
            entry["timestamp_seconds"] = start
            sep = "&" if "?" in meta["url"] else "?"
            entry["url"] = f"{meta['url']}{sep}t={start}s"
        citations.append(entry)
    return citations


def call_ollama(messages: List[dict], system: str, response_format: Optional[str] = None) -> str:
    """Call a local Ollama server's chat endpoint. Raises a clear error if Ollama isn't running."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }
    if response_format:
        payload["format"] = response_format

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Ollama. Make sure it's installed and running: "
            "run `ollama serve` (or just open the Ollama app), and that you've "
            f"pulled the model with `ollama pull {OLLAMA_MODEL}`."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' not found in Ollama. Pull it first: `ollama pull {OLLAMA_MODEL}`."
        )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def chat(session_id: str, user_message: str, document_ids: Optional[List[str]] = None, n_results: int = 5) -> dict:
    print("\n========== CHAT ==========")
    print("Message:", user_message)
    print("Document IDs:", document_ids)
    hits = vectorstore.query(user_message, n_results=n_results, document_ids=document_ids)
    context_str = format_context(hits) if hits else "No relevant context was found in the uploaded materials."

    # Replay prior turns (plain text, no injected context) so the model keeps conversational memory.
    history = db.get_history(session_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]

    messages.append({
        "role": "user",
        "content": f"Context excerpts:\n\n{context_str}\n\nQuestion: {user_message}",
    })

    answer = call_ollama(messages, system=SYSTEM_PROMPT)
    citations = build_citations(hits)

    db.add_message(session_id, "user", user_message)
    db.add_message(session_id, "assistant", answer, citations)

    return {"answer": answer, "citations": citations}
