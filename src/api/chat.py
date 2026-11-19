"""Chat API endpoints."""
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    SourceInfo,
    SocraticInitRequest,
    SocraticInitResponse,
    SocraticChatRequest,
    SocraticChatResponse
)
from src.agents import TeacherAgent, SocraticAgent
from src.services.socratic_session import SocraticSessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Keyed singletons — one agent instance per llm_provider value.
# This avoids re-initialising the client on every request while correctly
# routing "open_ai" and "self_host" calls to separate instances.
# ---------------------------------------------------------------------------

_agents: dict[str, TeacherAgent] = {}
_socratic_agents: dict[str, SocraticAgent] = {}
socratic_session_service = None


def get_agent(provider: str = "open_ai") -> TeacherAgent:
    """Get or create a TeacherAgent singleton for the given provider."""
    global _agents
    if provider not in _agents:
        _agents[provider] = TeacherAgent(llm_provider=provider)
    return _agents[provider]


def _format_sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_socratic_agent(provider: str = "open_ai") -> SocraticAgent:
    """Get or create a SocraticAgent singleton for the given provider."""
    global _socratic_agents
    if provider not in _socratic_agents:
        _socratic_agents[provider] = SocraticAgent(llm_provider=provider)
    return _socratic_agents[provider]


def get_socratic_session_service():
    """Get or initialize Socratic session singleton."""
    global socratic_session_service
    if socratic_session_service is None:
        socratic_session_service = SocraticSessionService()
    return socratic_session_service


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - Process user query and return response.
    
    - **query**: User's question or message (required)
    - **course_id**: Filter results by course ID (optional)
    - **module_id**: Filter results by module ID (optional)
    - **history**: Conversation history (optional)
    - **llm_provider**: LLM backend — `"open_ai"` (default) or `"self_host"` (vLLM)
    
    Returns:
    - **response**: Bot's answer
    - **intent**: Detected intent ('chat' or 'learning')
    - **sources**: Retrieved documents (only for 'learning' intent)
    """
    try:
        # Get or initialize agent for the requested provider
        chatbot = get_agent(request.llm_provider)
        
        # Convert history to dict format if provided
        history_list = None
        if request.history:
            try:
                history_list = [msg.model_dump() for msg in request.history]
            except Exception as e:
                # Log warning but continue without history
                logger.warning(f"Invalid history format, ignoring: {e}")
        
        # Process query with history
        result = chatbot.process(
            query=request.query,
            course_id=request.course_id,
            module_id=request.module_id,
            history=history_list
        )
        
        # Format sources
        sources = []
        if result.get("sources"):
            sources = [
                SourceInfo(
                    id=src.get("id", ""),
                    score=src.get("score", 0.0),
                    text=src.get("text", ""),
                    course_id=src.get("course_id"),
                    module_id=src.get("module_id"),
                    from_src=src.get("from_src")
                )
                for src in result["sources"]
            ]
        
        return ChatResponse(
            response=result["response"],
            intent=result.get("intent", "chat"),
            sources=sources,
            label=result.get("label", "learning"),
            refused=result.get("refused", False),
            refusal_reason=result.get("refusal_reason"),
            suggested_module_id=result.get("suggested_module_id"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/simple")
async def chat_simple(request: ChatRequest):
    """
    Simplified chat endpoint - Returns only the response text.
    
    Useful for simple integrations that only need the bot's answer.
    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    """
    try:
        chatbot = get_agent(request.llm_provider)
        result = chatbot.process(
            query=request.query,
            course_id=request.course_id,
            module_id=request.module_id
        )
        return {"response": result["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/socratic/init", response_model=SocraticInitResponse)
async def init_socratic_session(request: SocraticInitRequest):
    """Initialize Socratic session once with project document context."""
    try:
        session_service = get_socratic_session_service()

        problem_text = (request.problem or "").strip()
        if not problem_text and request.documents:
            documents_payload = [doc.model_dump() for doc in request.documents]
            problem_text = session_service.build_problem_from_documents(documents_payload)

        if not problem_text:
            raise HTTPException(
                status_code=400,
                detail="Missing problem context. Provide `problem` or at least one non-empty document content."
            )

        session = session_service.init_session(
            session_id=request.session_id,
            problem=problem_text,
            metadata=request.metadata,
            reset_history=request.reset_history
        )

        return SocraticInitResponse(
            session_id=request.session_id,
            initialized=True,
            has_problem=bool(session.get("problem")),
            problem_length=len(session.get("problem", ""))
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing Socratic session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing Socratic session: {str(e)}")


@router.post("/socratic", response_model=SocraticChatResponse)
async def socratic_chat(request: SocraticChatRequest):
    """Chat with Socratic agent using pre-initialized session state.

    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    """
    try:
        session_service = get_socratic_session_service()
        agent = get_socratic_agent(request.llm_provider)

        session = session_service.get_session(request.session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Session not found",
                    "message": "Please call /chat/socratic/init first when user enters project page"
                }
            )

        result = agent.process(
            query=request.query,
            problem=session.get("problem"),
            history=session.get("history", [])
        )

        response_text = result.get("response", "")
        session_service.append_turn(
            session_id=request.session_id,
            user_query=request.query,
            assistant_response=response_text
        )

        updated_session = session_service.get_session(request.session_id) or {}
        history_count = len(updated_session.get("history", []))

        return SocraticChatResponse(
            session_id=request.session_id,
            response=response_text,
            intent=result.get("intent", "chat"),
            history_count=history_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Socratic chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in Socratic chat: {str(e)}")


@router.delete("/socratic/session/{session_id}")
async def clear_socratic_session(session_id: str):
    """Clear one Socratic session (optional cleanup when leaving project page)."""
    try:
        session_service = get_socratic_session_service()
        deleted = session_service.clear_session(session_id)
        return {"session_id": session_id, "deleted": deleted}
    except Exception as e:
        logger.error(f"Error clearing Socratic session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing Socratic session: {str(e)}")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — SSE (Server-Sent Events).

    Same input as POST /chat.
    Supports **llm_provider** field (`"open_ai"` or `"self_host"`).
    Returns `text/event-stream` with:
      event: token   data: {"delta": "..."}
      event: done    data: {"intent": "...", "label": "...", "refused": bool,
                            "refusal_reason": "cross_module"|"out_of_corpus"|null,
                            "suggested_module_id": "..."|null, "sources": [...]}

    Refusals (label in cross_module/out_of_corpus) still stream as ordinary `token` events (a
    canned string, no LLM generation call) — NOT a new event type — so the refusal text keeps
    landing in whatever accumulates `token` deltas into the persisted/graded answer.
    """
    chatbot = get_agent(request.llm_provider)

    history_list = None
    if request.history:
        try:
            history_list = [msg.model_dump() for msg in request.history]
        except Exception as e:
            logger.warning(f"Invalid history format, ignoring: {e}")

    def event_generator():
        try:
            for event in chatbot.process_stream(
                query=request.query,
                course_id=request.course_id,
                module_id=request.module_id,
                history=history_list,
            ):
                if event["type"] == "token":
                    yield _format_sse("token", {"delta": event["delta"]})
                elif event["type"] == "done":
                    # Serialize sources to plain dicts
                    raw_sources = event.get("sources", [])
                    sources = []
                    for src in raw_sources:
                        if isinstance(src, dict):
                            sources.append({
                                "id": src.get("id", ""),
                                "score": src.get("score", 0.0),
                                "text": src.get("text", ""),
                                "course_id": src.get("course_id"),
                                "module_id": src.get("module_id"),
                                "from_src": src.get("from_src"),
                            })
                    yield _format_sse("done", {
                        "intent": event.get("intent", "chat"),
                        "label": event.get("label", "learning"),
                        "refused": event.get("refused", False),
                        "refusal_reason": event.get("refusal_reason"),
                        "suggested_module_id": event.get("suggested_module_id"),
                        "sources": sources,
                    })
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
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
