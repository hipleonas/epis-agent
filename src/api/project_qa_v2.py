"""Project QA v2 API.

Endpoints:
- POST /api/v2/project-qa/sessions
- GET  /api/v2/project-qa/sessions/{session_id}
- POST /api/v2/project-qa/sessions/{session_id}/start-interview
- POST /api/v2/project-qa/sessions/{session_id}/respond
- GET  /api/v2/project-qa/sessions/{session_id}/report
"""

import logging
from threading import Lock
from typing import Any, Literal, Optional, TypedDict
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.agents.project_qa_agent import (
    SUBMISSION_VALIDITY_THRESHOLD,
    VERDICT_LABELS,
    ChatTurn,
    GradingReport,
    ProjectQAOrchestrator,
    Question,
    StudentSubmission,
    SubmissionValidationResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/project-qa", tags=["project-qa-v2"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    project_description: str = Field(..., min_length=1)
    rubric: Optional[str] = None
    student_name: Optional[str] = None
    student_submission: Optional[StudentSubmission] = None
    llm_provider: Literal["open_ai", "self_host"] = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )
    validate_submission: bool = Field(
        True,
        description=(
            "Bật/tắt bước đối chiếu bài nộp với đề bài. Caller PHẢI gửi false khi "
            "project_description không phải đề bài thật (ví dụ nó được tóm tắt ra từ chính "
            "bài nộp) — nếu không, bài nộp bị đem so với mô tả của chính nó và luôn hợp lệ."
        ),
    )


class SubmissionValidationResponse(BaseModel):
    """Kết quả đối chiếu bài nộp với đề bài, hiển thị được cho học sinh."""

    status: Literal["valid", "invalid", "unknown"]
    is_valid: bool
    validity_score: int = Field(..., ge=0, le=100)
    threshold: int = SUBMISSION_VALIDITY_THRESHOLD
    verdict_label: str
    issues: list[str] = []
    summary: str = ""


def _to_validation_response(validation: SubmissionValidationResult) -> SubmissionValidationResponse:
    status = validation.get("status", "unknown")
    return SubmissionValidationResponse(
        status=status,
        is_valid=status == "valid",
        validity_score=int(validation.get("validity_score", 0)),
        threshold=SUBMISSION_VALIDITY_THRESHOLD,
        verdict_label=validation.get("verdict_label") or VERDICT_LABELS[status],
        issues=list(validation.get("issues", [])),
        summary=str(validation.get("summary") or ""),
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    questions: list[Question]
    status: str
    submission_validation: SubmissionValidationResponse


class StartInterviewResponse(BaseModel):
    question_index: int
    agent_message: str
    current_question_id: str
    submission_validation: SubmissionValidationResponse


class StudentRespondRequest(BaseModel):
    student_message: str = Field(..., min_length=1)


class StudentRespondResponse(BaseModel):
    agent_message: str
    current_question_id: str
    question_index: int
    question_status: str
    interview_status: str
    hint_count: int = 0
    hints_remaining: int = 2


class SessionStatusResponse(BaseModel):
    session_id: str
    stage: str
    interview_status: str
    grading_status: str
    current_question_index: int
    current_question_id: Optional[str]
    questions: list[Question]
    total_turns: int
    submission_validation: SubmissionValidationResponse
    grading_report: Optional[GradingReport] = None
    pipeline_error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory Session Storage
# ---------------------------------------------------------------------------

class SessionData(TypedDict):
    session_id: str
    project_description: str
    rubric: Optional[str]
    student_name: Optional[str]
    student_submission: Optional[StudentSubmission]
    submission_validation: SubmissionValidationResult
    questions: list[Question]
    stage: str
    interview_status: str
    grading_status: str
    current_question_index: int
    current_question_id: Optional[str]
    full_chat_log: list[ChatTurn]
    last_agent_message: str
    grading_report: Optional[GradingReport]
    pipeline_error: Optional[str]
    grading_task_queued: bool
    llm_provider: str  # Stored at session creation; inherited by interview/grading steps


# Keyed orchestrators — one ProjectQAOrchestrator per llm_provider.
_orchestrators: dict[str, ProjectQAOrchestrator] = {}
_sessions: dict[str, SessionData] = {}
_sessions_lock = Lock()


def _get_orchestrator(provider: str = "open_ai") -> ProjectQAOrchestrator:
    global _orchestrators
    if provider not in _orchestrators:
        _orchestrators[provider] = ProjectQAOrchestrator(llm_provider=provider)
    return _orchestrators[provider]


def _neutral_validation() -> SubmissionValidationResult:
    """Trạng thái mặc định khi chưa/không xác thực được — KHÔNG phải 'hợp lệ'."""
    return {
        "is_valid": False,
        "validity_score": 0,
        "status": "unknown",
        "verdict_label": VERDICT_LABELS["unknown"],
        "issues": [],
        "summary": "",
    }


def _get_session_or_404(session_id: str) -> SessionData:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _run_grading_for_session(session_id: str) -> None:
    try:
        with _sessions_lock:
            session = _sessions.get(session_id)
            if not session:
                return
            provider = session.get("llm_provider", "open_ai")

        orchestrator = _get_orchestrator(provider)

        with _sessions_lock:
            session = _sessions.get(session_id)
            if not session:
                return
            if session["grading_status"] == "completed":
                session["grading_task_queued"] = False
                return
            session["grading_task_queued"] = False
            session["stage"] = "grading"
            session["grading_status"] = "in_progress"
            session["pipeline_error"] = None

        validation = session.get("submission_validation") or _neutral_validation()
        report = orchestrator.grade(
            questions=session["questions"],
            chat_log=session["full_chat_log"],
            rubric=session["rubric"],
            student_name=session["student_name"],
            student_submission=session.get("student_submission"),
            submission_status=validation["status"],
            submission_validity_score=validation["validity_score"],
            submission_issues=validation["issues"],
            submission_summary=validation["summary"],
        )

        with _sessions_lock:
            latest = _sessions.get(session_id)
            if not latest:
                return
            latest["grading_report"] = report
            latest["grading_status"] = "completed"
            latest["stage"] = "done"
            latest["grading_task_queued"] = False
    except Exception as exc:
        logger.error("Background grading failed for session %s: %s", session_id, exc, exc_info=True)
        with _sessions_lock:
            latest = _sessions.get(session_id)
            if not latest:
                return
            latest["grading_status"] = "failed"
            latest["stage"] = "done"
            latest["pipeline_error"] = str(exc)
            latest["grading_task_queued"] = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_project_qa_session(request: CreateSessionRequest):
    """Create a new session and run Agent 1 (question generation).

    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    The provider is stored in the session and automatically inherited
    by subsequent /start-interview and /respond calls.
    """
    orchestrator = _get_orchestrator(request.llm_provider)
    try:
        questions, validation_result = orchestrator.generate_questions(
            project_description=request.project_description,
            rubric=request.rubric,
            student_submission=request.student_submission,
            validate_submission=request.validate_submission,
        )
    except Exception as exc:
        logger.error("Failed to create session: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {exc}")

    session_id = str(uuid4())
    session_data: SessionData = {
        "session_id": session_id,
        "project_description": request.project_description,
        "rubric": request.rubric,
        "student_name": request.student_name,
        "student_submission": request.student_submission,
        "submission_validation": validation_result,
        "questions": questions,
        "stage": "questioning",
        "interview_status": "not_started",
        "grading_status": "pending",
        "current_question_index": 0,
        "current_question_id": questions[0]["question_id"] if questions else None,
        "full_chat_log": [],
        "last_agent_message": "",
        "grading_report": None,
        "pipeline_error": None,
        "grading_task_queued": False,
        "llm_provider": request.llm_provider,
    }

    with _sessions_lock:
        _sessions[session_id] = session_data

    return CreateSessionResponse(
        session_id=session_id,
        questions=questions,
        status="ready",
        submission_validation=_to_validation_response(validation_result),
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_project_qa_session_status(session_id: str):
    """Get current session state (stage/current question/report if ready)."""
    session = _get_session_or_404(session_id)
    return SessionStatusResponse(
        session_id=session["session_id"],
        stage=session["stage"],
        interview_status=session["interview_status"],
        grading_status=session["grading_status"],
        current_question_index=session["current_question_index"],
        current_question_id=session["current_question_id"],
        questions=session["questions"],
        total_turns=len(session["full_chat_log"]),
        submission_validation=_to_validation_response(
            session.get("submission_validation") or _neutral_validation()
        ),
        grading_report=session["grading_report"],
        pipeline_error=session["pipeline_error"],
    )


@router.post("/sessions/{session_id}/start-interview", response_model=StartInterviewResponse)
async def start_interview(session_id: str):
    """Start Agent 2 and return first agent question."""
    session = _get_session_or_404(session_id)

    if session["interview_status"] == "completed":
        raise HTTPException(status_code=409, detail="Interview has already completed")

    # Resolve orchestrator from the provider stored at session creation time.
    provider = session.get("llm_provider", "open_ai")
    orchestrator = _get_orchestrator(provider)

    validation = session.get("submission_validation") or _neutral_validation()

    try:
        if session["interview_status"] == "not_started":
            state = orchestrator.start_interview_session(
                session_id=session_id,
                questions=session["questions"],
                student_name=session["student_name"],
                student_submission=session.get("student_submission"),
                submission_status=validation["status"],
                submission_validity_score=validation["validity_score"],
                submission_issues=validation["issues"],
                submission_summary=validation["summary"],
            )
        else:
            state = orchestrator.get_interview_session_state(session_id)

        with _sessions_lock:
            latest = _sessions[session_id]
            latest["stage"] = "interview"
            latest["interview_status"] = state.get("interview_status", "in_progress")
            latest["current_question_index"] = state.get("current_question_index", 0)
            latest["current_question_id"] = state.get("current_question_id")
            latest["full_chat_log"] = state.get("full_chat_log", [])
            latest["last_agent_message"] = state.get("last_agent_message", "")
            latest["pipeline_error"] = state.get("error_message")

        return StartInterviewResponse(
            question_index=state.get("current_question_index", 0),
            agent_message=state.get("last_agent_message", ""),
            current_question_id=state.get("current_question_id", ""),
            submission_validation=_to_validation_response(validation),
        )
    except Exception as exc:
        logger.error("Failed to start interview for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {exc}")


@router.post("/sessions/{session_id}/respond", response_model=StudentRespondResponse)
async def respond_to_interview(
    session_id: str,
    request: StudentRespondRequest,
    background_tasks: BackgroundTasks,
):
    """Send student answer and get next agent response from Agent 2."""
    session = _get_session_or_404(session_id)

    if session["interview_status"] == "not_started":
        raise HTTPException(status_code=409, detail="Interview has not started. Call /start-interview first")
    if session["interview_status"] == "completed":
        raise HTTPException(status_code=409, detail="Interview already completed")

    # Resolve orchestrator from the provider stored at session creation time.
    provider = session.get("llm_provider", "open_ai")
    orchestrator = _get_orchestrator(provider)

    try:
        state = orchestrator.resume_interview_session(
            session_id=session_id,
            student_message=request.student_message,
        )

        interview_status = state.get("interview_status", "in_progress")
        question_status = state.get("current_question_status", "probing")

        with _sessions_lock:
            latest = _sessions[session_id]
            latest["stage"] = "interview" if interview_status != "completed" else "grading"
            latest["interview_status"] = interview_status
            latest["current_question_index"] = state.get("current_question_index", 0)
            latest["current_question_id"] = state.get("current_question_id")
            latest["full_chat_log"] = state.get("full_chat_log", [])
            latest["last_agent_message"] = state.get("last_agent_message", "")
            latest["pipeline_error"] = state.get("error_message")

            if interview_status == "completed" and latest["grading_status"] == "pending" and not latest.get("grading_task_queued", False):
                latest["grading_task_queued"] = True
                latest["stage"] = "grading"
                background_tasks.add_task(_run_grading_for_session, session_id)

        return StudentRespondResponse(
            agent_message=state.get("last_agent_message", ""),
            current_question_id=state.get("current_question_id", ""),
            question_index=state.get("current_question_index", 0),
            question_status=question_status,
            interview_status=interview_status,
            hint_count=state.get("hint_count", 0),
            hints_remaining=max(
                0,
                state.get("max_hints_per_question", 2) - state.get("hint_count", 0),
            ),
        )
    except Exception as exc:
        logger.error("Failed to respond session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process response: {exc}")


@router.get("/sessions/{session_id}/report", response_model=GradingReport)
async def get_grading_report(session_id: str, background_tasks: BackgroundTasks):
    """Return final grading report after interview completion."""
    session = _get_session_or_404(session_id)
    logger.info("Fetching grading report for session %s: interview_status=%s, grading_status=%s", session_id, session["interview_status"], session["grading_status"])

    if session["grading_report"] is not None:
        return session["grading_report"]

    if session["interview_status"] != "completed":
        logger.warning("Attempting to fetch grading report for incomplete interview session %s", session_id)
        raise HTTPException(status_code=409, detail="Interview chưa hoàn thành")

    if session["grading_status"] == "failed":
        logger.error("Failed to generate grading report for session %s: %s", session_id, session["pipeline_error"])
        raise HTTPException(
            status_code=500,
            detail=session["pipeline_error"] or "Chấm điểm thất bại",
        )

    if session["grading_status"] in {"pending", "in_progress"}:
        with _sessions_lock:
            latest = _sessions.get(session_id)
            if latest and latest["grading_report"] is None and latest["grading_status"] != "completed":
                if not latest.get("grading_task_queued", False):
                    # Self-heal stale in_progress sessions by allowing worker to restart.
                    if latest["grading_status"] == "in_progress":
                        logger.warning("Detected stale in_progress grading for session %s. Re-queue grading worker.", session_id)
                        latest["grading_status"] = "pending"
                    latest["grading_task_queued"] = True
                    latest["stage"] = "grading"
                    background_tasks.add_task(_run_grading_for_session, session_id)
        logger.info("Grading report for session %s is in progress", session_id)
        raise HTTPException(status_code=202, detail="Đang chấm điểm, vui lòng thử lại sau 2 giây")

    logger.error("Unexpected grading status for session %s: %s", session_id, session["grading_status"])
    raise HTTPException(status_code=500, detail=session["pipeline_error"] or "Báo cáo chưa có")


@router.delete("/sessions/{session_id}")
async def delete_project_qa_session(session_id: str) -> dict[str, Any]:
    """Xoá một phiên vấn đáp (dùng khi học sinh bấm vấn đáp lại).

    Phiên vấn đáp và checkpoint LangGraph đều nằm trong RAM của tiến trình. Nếu retake chỉ
    tạo phiên mới mà không xoá phiên cũ thì mỗi lần làm lại để rớt lại một phiên chết.
    Idempotent: xoá phiên không tồn tại vẫn trả 200.
    """
    with _sessions_lock:
        session = _sessions.pop(session_id, None)

    if session:
        provider = session.get("llm_provider", "open_ai")
        try:
            _get_orchestrator(provider).drop_interview_session(session_id)
        except Exception as exc:
            logger.warning("Failed to drop interview checkpoint %s: %s", session_id, exc)

    return {"deleted": bool(session), "session_id": session_id}


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    """Lightweight debug endpoint for in-memory sessions."""
    with _sessions_lock:
        data = [
            {
                "session_id": s["session_id"],
                "stage": s["stage"],
                "interview_status": s["interview_status"],
                "grading_status": s["grading_status"],
            }
            for s in _sessions.values()
        ]
    return {"count": len(data), "items": data}
