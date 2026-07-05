"""
Thin wrapper around ChromaDB. Uses Chroma's bundled local ONNX MiniLM
embedding function, so no external embedding API / API key / GPU is needed.
"""
import os
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION_NAME = "study_assistant"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = embedding_functions.DefaultEmbeddingFunction()  # local, free, fast
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    return _collection


def add_records(records: List[Dict]):
    if not records:
        return
    collection = get_collection()
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )


def query(query_text: str, n_results: int = 5, document_ids: Optional[List[str]] = None) -> List[Dict]:
    collection = get_collection()
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    hits = []
    if not results["ids"] or not results["ids"][0]:
        return hits
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return hits


def get_texts_for_documents(document_ids: Optional[List[str]] = None, limit: int = 100) -> List[str]:
    collection = get_collection()
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    data = collection.get(where=where, limit=limit)
    return data.get("documents", []) or []


def delete_document(document_id: str):
    collection = get_collection()
    collection.delete(where={"document_id": document_id})
