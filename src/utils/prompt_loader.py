"""Shared prompt-file loader for all agents.

Convention: prompt text lives in `src/agents/prompt/<agent>/<file_name>` as plain UTF-8 text, so
prompts can be tuned without touching Python code. If the file is missing (fresh checkout, ops
forgot to sync it, etc.) callers get a safe fallback instead of crashing.

Lives under `src/utils/` (not `src/agents/`) on purpose — `src/agents/__init__.py` eagerly imports
`teacher_agent`, which imports `src.services.rag`, which needs this loader too. Putting the loader
inside `src.agents` would make that import re-enter `src/agents/__init__.py` mid-load and crash
with a circular-import error.
"""
from __future__ import annotations

from pathlib import Path

# src/utils/prompt_loader.py -> parent.parent = src/ -> src/agents/prompt/<agent>/<file>
_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "agents" / "prompt"


def read_prompt(agent: str, file_name: str, fallback: str = "") -> str:
    """Read `src/agents/prompt/<agent>/<file_name>`, stripped. Returns `fallback` if missing."""
    path = _PROMPT_ROOT / agent / file_name
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return fallback
