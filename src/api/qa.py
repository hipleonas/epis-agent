"""QA (Oral Exam) API endpoints.

POST /qa/start          → Open a session, return greeting + first question.
POST /qa/answer         → Process one student turn, return response + scoring metadata.
POST /qa/start/stream   → Same as /qa/start but via SSE.
POST /qa/answer/stream  → Same as /qa/answer but via SSE.
"""
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.models.schemas import (
    QAStartRequest,
    QAStartResponse,
    QAAnswerRequest,
    QAAnswerResponse,
    ScoringEvent,
)
from src.services.qa_agent import QAAgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["oral-exam"])

# Keyed singletons — one QAAgentService per llm_provider.
_qa_agents: dict[str, QAAgentService] = {}


def _get_agent(provider: str = "open_ai") -> QAAgentService:
    global _qa_agents
    if provider not in _qa_agents:
        _qa_agents[provider] = QAAgentService(llm_provider=provider)
    return _qa_agents[provider]


def _format_sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# POST /qa/start
# ---------------------------------------------------------------------------

@router.post("/start", response_model=QAStartResponse)
async def start_exam_session(request: QAStartRequest):
    """Start an oral exam session (non-streaming).

    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    """
    try:
        agent = _get_agent(request.llm_provider)
        response_text = agent.start_session(
            question_bank=[q.model_dump() for q in request.question_bank],
            project_summary=request.project_summary,
            max_questions=request.max_questions,
        )
        return QAStartResponse(response=response_text)
    except Exception as exc:
        logger.error("POST /qa/start failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start session: {exc}")


@router.post("/start/stream")
async def start_exam_session_stream(request: QAStartRequest):
    """Start an oral exam session with SSE streaming.

    Same input as POST /qa/start.
    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    Returns `text/event-stream` with:
      event: token   data: {"delta": "..."}
      event: done    data: {}
    """
    agent = _get_agent(request.llm_provider)

    def event_generator():
        try:
            for event in agent.start_session_stream(
                question_bank=[q.model_dump() for q in request.question_bank],
                project_summary=request.project_summary,
                max_questions=request.max_questions,
            ):
                if event["type"] == "token":
                    yield _format_sse("token", {"delta": event["delta"]})
                elif event["type"] == "done":
                    yield _format_sse("done", {})
        except Exception as exc:
            logger.error("Stream /qa/start failed: %s", exc, exc_info=True)
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /qa/answer
# ---------------------------------------------------------------------------

@router.post("/answer", response_model=QAAnswerResponse)
async def answer_turn(request: QAAnswerRequest):
    """Process one student turn during the oral exam (non-streaming).

    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    """
    try:
        agent = _get_agent(request.llm_provider)
        result = agent.process_answer(
            user_message=request.user_message,
            current_question=request.current_question.model_dump(),
            current_question_turns=request.current_question_turns,
            score_locked=request.score_locked.model_dump() if request.score_locked else None,
            project_summary=request.project_summary,
            history=request.history,
            is_last_question=request.is_last_question,
        )

        scoring_event = None
        if result.get("scoring_event"):
            se = result["scoring_event"]
            scoring_event = ScoringEvent(
                question_id=se["question_id"],
                score=se["score"],
                max_score=se["max_score"],
            )

        return QAAnswerResponse(
            response=result["response"],
            scoring_event=scoring_event,
            advance_to_next=result.get("advance_to_next", False),
            is_complete=result.get("is_complete", False),
        )
    except Exception as exc:
        logger.error("POST /qa/answer failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {exc}")


@router.post("/answer/stream")
async def answer_turn_stream(request: QAAnswerRequest):
    """Process one student turn with SSE streaming.

    Same input as POST /qa/answer.
    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    Returns `text/event-stream` with:
      event: token   data: {"delta": "..."}
      event: done    data: {"scoring_event": {...}|null, "advance_to_next": bool, "is_complete": bool}
    """
    agent = _get_agent(request.llm_provider)

    def event_generator():
        try:
            for event in agent.process_answer_stream(
                user_message=request.user_message,
                current_question=request.current_question.model_dump(),
                current_question_turns=request.current_question_turns,
                score_locked=request.score_locked.model_dump() if request.score_locked else None,
                project_summary=request.project_summary,
                history=request.history,
                is_last_question=request.is_last_question,
            ):
                if event["type"] == "token":
                    yield _format_sse("token", {"delta": event["delta"]})
                elif event["type"] == "done":
                    done_data = {
                        "scoring_event": event.get("scoring_event"),
                        "advance_to_next": event.get("advance_to_next", False),
                        "is_complete": event.get("is_complete", False),
                    }
                    yield _format_sse("done", done_data)
        except Exception as exc:
            logger.error("Stream /qa/answer failed: %s", exc, exc_info=True)
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
