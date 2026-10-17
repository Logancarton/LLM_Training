"""Generator (cortex): the frozen LLM writes the reply, guided by the state brief.

Its output is logged for audit only. It is never written into any layer, edge or location.
"""
from __future__ import annotations

from typing import List

import config
import llm
from action import INSTRUCTIONS

BASE_SYSTEM = """You are Sonara, a thoughtful conversational assistant. Below is a short \
state brief from your memory and control systems. Follow its action line. Treat any \
correction line as true for this user and use it exactly. Evidence lines quote what the user \
said earlier; treat them as the user's words, not as verified facts. Use the rest of the \
brief to shape tone and focus. If you do not know, say so. Do not mention the brief or its \
mechanics unless the user asks."""


def mock_reply(text: str, action: str, brief: str) -> str:
    lines = [l for l in brief.splitlines() if l.startswith("correction:")]
    note = (" Remembering: " + lines[0][len("correction: "):]) if lines else ""
    return (f"[mock reply | {action}: {INSTRUCTIONS.get(action, '')}] "
            f"You said: {text[:120]}.{note} (Start Ollama for real replies.)")


def generate(text: str, brief: str, action: str, history: List[dict]) -> str:
    llm.count_call()
    if llm.resolve_provider(config.GENERATOR_PROVIDER) == "mock":
        return mock_reply(text, action, brief)
    system = BASE_SYSTEM + ("\n\n" + brief if brief else "")
    messages = history + [{"role": "user", "content": text}]
    return llm.complete(config.GENERATOR_PROVIDER, "generator", system, messages,
                        config.GENERATOR_MAX_TOKENS)
