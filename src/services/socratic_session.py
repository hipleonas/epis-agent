"""In-memory session store for Socratic chat state."""
import os
import time
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from pypdf import PdfReader


logger = logging.getLogger(__name__)


class SocraticSessionService:
    """Manage Socratic chat sessions with preloaded problem context."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        self._ttl_seconds = int(os.getenv("SOCRATIC_SESSION_TTL_SECONDS", "7200"))

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if (now - session.get("updated_at", now)) > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _now(self) -> float:
        return time.time()

    def init_session(
        self,
        session_id: str,
        problem: str,
        metadata: Optional[Dict[str, Any]] = None,
        reset_history: bool = True
    ) -> Dict[str, Any]:
        with self._lock:
            self._cleanup_expired()

            existing = self._sessions.get(session_id, {})
            history = [] if reset_history else existing.get("history", [])

            self._sessions[session_id] = {
                "problem": problem,
                "metadata": metadata or {},
                "history": history,
                "created_at": existing.get("created_at", self._now()),
                "updated_at": self._now()
            }
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup_expired()
            session = self._sessions.get(session_id)
            if not session:
                return None
            session["updated_at"] = self._now()
            return session

    def append_turn(self, session_id: str, user_query: str, assistant_response: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            session.setdefault("history", []).append({"role": "user", "content": user_query})
            session.setdefault("history", []).append({"role": "assistant", "content": assistant_response})
            session["updated_at"] = self._now()

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _read_text_from_file(self, file_path: str, doc_type: str) -> str:
        """Read and extract text content from local file path."""
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists() or not path.is_file():
            logger.warning("Socratic document file not found: %s", path)
            return ""

        resolved_type = (doc_type or "").lower()
        if not resolved_type:
            resolved_type = "pdf" if path.suffix.lower() == ".pdf" else "markdown"

        try:
            if resolved_type == "pdf":
                reader = PdfReader(str(path))
                page_texts: List[str] = []
                for page in reader.pages:
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        page_texts.append(page_text)
                return "\n\n".join(page_texts).strip()

            # Default: treat as plain text/markdown
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as exc:
            logger.warning("Failed to read Socratic document file %s: %s", path, exc)
            return ""

    def build_problem_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Convert list of markdown/pdf documents into one problem context string."""
        parts: List[str] = []
        for idx, doc in enumerate(documents, 1):
            doc_type = (doc.get("type") or "unknown").lower()
            title = doc.get("title") or f"document_{idx}"
            content = (doc.get("content") or "").strip()

            if not content and doc.get("file_path"):
                content = self._read_text_from_file(doc.get("file_path"), doc_type)

            if not content:
                continue
            parts.append(f"[{doc_type.upper()}] {title}\n{content}")

        return "\n\n".join(parts).strip()
