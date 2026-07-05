"""
Quiz generation: pulls chunk text for the chosen document(s) and asks a
local LLM (via Ollama) to produce a structured JSON quiz (MCQ + short answer).
"""
import json
from typing import List, Optional

from . import vectorstore
from .rag import call_ollama

QUIZ_SYSTEM_PROMPT = """You create quizzes from study material for exam prep. Generate ONLY multiple-choice questions.
Return ONLY valid JSON (no markdown fences, no commentary) matching exactly:
{
  "questions": [
    {
      "question": "string",
      "type": "mcq",
      "options": [
        "string",
        "string",
        "string",
        "string"
      ],
      "answer": "string",
      "explanation": "string"
    }
  ]
}

Rules:

- Every question MUST have exactly four options.
- Exactly one option must be correct.
- The "type" field MUST always be "mcq".
- Never use "short_answer".
- Return ONLY JSON."""


def generate_quiz(document_ids: Optional[List[str]] = None, num_questions: int = 5, difficulty: str = "medium") -> dict:
    texts = vectorstore.get_texts_for_documents(document_ids=document_ids, limit=8)
    if not texts:
        return {"questions": []}

    combined = "\n\n".join(texts)
    # Keep the prompt bounded regardless of how much material was uploaded.
    combined = combined[:5000]

    user_prompt = (
        f"Generate {num_questions} quiz questions at {difficulty} difficulty from this study material. "
        f"Generate only multiple-choice questions, Each question must have exactly four options.Return valid JSON only. "
        f"repeating one section.\n\nStudy material:\n{combined}"
    )

    raw = call_ollama(
        messages=[{"role": "user", "content": user_prompt}],
        system=QUIZ_SYSTEM_PROMPT,
        response_format="json",
    )

    print("OLLAMA RESPONSE:")
    print(raw)
    print(type(raw))
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"questions": [], "error": "Could not parse quiz JSON from model output", "raw": raw}
