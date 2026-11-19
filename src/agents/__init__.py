"""Agents package for chatbot agents."""
from .teacher_agent import TeacherAgent
from .socratic_agent import SocraticAgent

try:
	from .detective_agent import DetectiveAgent  # type: ignore
except Exception:
	DetectiveAgent = None

__all__ = ['TeacherAgent', 'SocraticAgent', 'DetectiveAgent']
