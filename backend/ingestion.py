"""
Ingestion pipeline: turns PDFs and YouTube URLs into text chunks with
retrieval-friendly metadata (page number for PDFs, timestamp for YouTube).
"""
import re
from typing import List, Dict, Optional

from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks (keeps context continuous)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window character chunker. Good enough for note-style PDFs."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def process_pdf(filepath: str, doc_id: str, doc_name: str) -> List[Dict]:
    """Extract text per page, chunk it, and tag each chunk with its page number."""
    reader = PdfReader(filepath)
    records = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "id": f"{doc_id}_p{page_num}_c{i}",
                "text": chunk,
                "metadata": {
                    "document_id": doc_id,
                    "source_name": doc_name,
                    "source_type": "pdf",
                    "page": page_num,
                },
            })
    return records


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract a YouTube video ID from that URL.")


def process_youtube(
    url: str,
    doc_id: str,
    doc_name: str,
    seconds_per_chunk: int = 45,
) -> List[Dict]:
    """
    Pull the transcript (no audio download / no Whisper needed - fast + free)
    and group cues into ~45s chunks, tagging each with its start timestamp.
    """
    video_id = extract_video_id(url)
    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(video_id)

    records = []
    buffer: List[str] = []
    chunk_start: Optional[float] = None
    chunk_idx = 0

    def flush():
        nonlocal buffer, chunk_start, chunk_idx
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if text:
            records.append({
                "id": f"{doc_id}_t{int(chunk_start)}_c{chunk_idx}",
                "text": text,
                "metadata": {
                    "document_id": doc_id,
                    "source_name": doc_name,
                    "source_type": "youtube",
                    "video_id": video_id,
                    "url": url,
                    "start_time": float(chunk_start),
                },
            })
            chunk_idx += 1
        buffer = []
        chunk_start = None

    for entry in transcript:

        if chunk_start is None:
            chunk_start = entry.start

        buffer.append(entry.text)

        if entry.start - chunk_start >= seconds_per_chunk:
            flush()

    return records
