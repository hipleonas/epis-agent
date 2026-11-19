"""API package for FastAPI routers."""
from .chat import router as chat_router
from .upload import router as upload_router
from .github import router as github_router
from .qa import router as qa_router
from .analyze import router as analyze_router

__all__ = ['chat_router', 'upload_router', 'github_router', 'qa_router', 'analyze_router']
