"""FastAPI OpenAI-compatible chat completions wrapping Bedrock Kimi K2.5.

s3 paper's generator_llms/local_inst.py posts OpenAI-format chat completion
requests to http://...:8000/v1/chat/completions and expects a standard
{"choices": [{"message": {"content": "..."}, ...}], ...} response.

This adapter exposes the same surface but routes through Bedrock instead of
vLLM. The paper's §3 modular design explicitly supports this — "compatible
with frozen or proprietary models" (see paper abstract).

Run:
    uvicorn finqa_generator_adapter:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

REPO_ROOT = Path(os.getenv(
    "FINQA_REPO_ROOT",
    "/home/ubuntu/fin-qa-research"
))
sys.path.insert(0, str(REPO_ROOT / "src" / "finqa_common" / "src"))

from finqa_common.utils import generate_answer, get_bedrock_client  # noqa: E402


GENERATOR_MODEL = os.getenv("FINQA_GENERATOR_MODEL", "moonshotai.kimi-k2.5")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_MAX_TOKENS = int(os.getenv("FINQA_GEN_MAX_TOKENS", "2048"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


app = FastAPI()
_bedrock_client = None


def _ensure_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = get_bedrock_client(region=AWS_REGION)


def _flatten_messages(messages: List[ChatMessage]) -> tuple[str, str]:
    """Bedrock generate_answer takes (user_prompt, system_prompt). Concatenate
    multi-turn user/assistant history into a single user prompt."""
    system_parts: List[str] = []
    convo_parts: List[str] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            convo_parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            convo_parts.append(f"Assistant: {m.content}")
    return "\n\n".join(convo_parts), "\n\n".join(system_parts)


@app.on_event("startup")
def _startup() -> None:
    _ensure_client()


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="empty messages")
    _ensure_client()
    user_prompt, system_prompt = _flatten_messages(request.messages)
    text = generate_answer(
        user_prompt,
        GENERATOR_MODEL,
        _bedrock_client,
        request.max_tokens or DEFAULT_MAX_TOKENS,
        request.temperature,
        system_prompt=system_prompt or None,
    )
    text = text or ""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or GENERATOR_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": GENERATOR_MODEL}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": GENERATOR_MODEL, "object": "model", "owned_by": "bedrock"}],
    }


if __name__ == "__main__":
    uvicorn.run(
        "finqa_generator_adapter:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
    )
