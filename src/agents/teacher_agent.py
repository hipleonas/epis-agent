"""LangGraph-based agent for routing between normal chat and RAG (retrieve-first)."""
import os
from typing import TypedDict, Literal, Optional, Dict, List, Generator
from openai import OpenAI
from langgraph.graph import StateGraph, END
from src.services.rag import RAGService
from src.services.llm_client_factory import build_llm_client
from src.agents.refusals import REFUSAL_OUT_OF_CORPUS
from src.utils.prompt_loader import read_prompt

_DEFAULT_CHAT_SYSTEM_PROMPT = read_prompt(
    "teacher_agent", "chat_system_prompt.txt",
    """Bạn là một trợ lý ảo thân thiện và hữu ích cho một nền tảng học trực tuyến.
Bạn có thể:
- Trả lời các câu hỏi chung
- Hướng dẫn học sinh cách sử dụng hệ thống
- Động viên và khuyến khích học sinh
- Trả lời bằng tiếng Việt một cách thân thiện
- Không trả lời các câu hỏi không liên quan đến việc học
- Từ chối trả lời các thông tin bên ngoài

Nếu học sinh hỏi về nội dung bài học cụ thể, hãy khuyến khích họ nói rõ module hoặc khóa học họ đang học.""",
)

# Chỉ CÒN out_of_corpus là từ chối. `cross_module` giờ vẫn trả lời bình thường (ngữ cảnh gồm cả
# chunk của module chứa nội dung), rồi server nối câu gợi ý mở module đó vào cuối câu trả lời.
_REFUSAL_LABELS = ("out_of_corpus",)


class AgentState(TypedDict):
    """State for the agent graph."""
    query: str
    intent: Optional[str]              # "chat" | "learning" — hình chiếu 2 giá trị, giữ tương thích ngược
    label: Optional[str]               # "chat" | "learning" | "cross_module" | "out_of_corpus"
    suggested_module_id: Optional[str]
    docs: Optional[List[Dict]]         # chunk module hiện tại đã lọc, dùng để sinh câu trả lời
    course_id: Optional[str]
    module_id: Optional[str]
    history: Optional[list]  # Conversation history
    response: Optional[str]
    sources: Optional[list]


class TeacherAgent:
    """LangGraph agent that routes between normal chat, RAG-based learning, and refusal."""

    def __init__(self, llm_provider: str = "open_ai"):
        """Initialize the agent with LLM and RAG service."""
        self.llm_client, self.model = build_llm_client(llm_provider)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        # Truyền provider xuống RAGService để bộ phân loại retrieve-first và bước sinh câu trả lời
        # dùng CÙNG một backend LLM với agent — không lẫn open_ai với self_host trong một lượt chat.
        self.rag_service = RAGService(llm_provider=llm_provider)
        self.graph = self._build_graph()

    def _retrieve_and_classify(self, state: AgentState) -> AgentState:
        """Truy xuất trước, rồi phân loại 4 nhãn dựa trên chính các chunk truy xuất được.

        Đây là node DUY NHẤT quyết định định tuyến — cả `process()` (graph) lẫn `process_stream()`
        (SSE, không dùng graph) đều gọi đúng hàm này, nên hai luồng không thể lệch nhau.
        """
        result = self.rag_service.analyze(
            query=state["query"],
            course_id=state.get("course_id"),
            module_id=state.get("module_id"),
            history=state.get("history", []),
        )
        state["label"] = result["label"]
        state["intent"] = result["intent"]
        state["suggested_module_id"] = result["suggested_module_id"]
        state["docs"] = result["docs"]
        return state

    def _handle_normal_chat(self, state: AgentState) -> AgentState:
        """Handle normal chat without RAG."""
        query = state["query"]
        history = state.get("history", [])

        messages = self._build_chat_messages(query, history)

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        state["response"] = response.choices[0].message.content
        state["sources"] = []

        return state

    def _handle_learning_query(self, state: AgentState) -> AgentState:
        """Handle learning-related query using RAG — dùng chunk đã truy xuất sẵn, không truy xuất lại."""
        result = self.rag_service.answer_question(
            query=state["query"],
            context_docs=state.get("docs") or [],
            history=state.get("history", []),
        )

        state["response"] = result["answer"]
        state["sources"] = result["sources"]

        return state

    def _handle_refusal(self, state: AgentState) -> AgentState:
        """Từ chối CỐ ĐỊNH cho out_of_corpus. KHÔNG gọi LLM sinh câu trả lời — đây là chốt chặn
        "zero rò rỉ": không có nội dung nào trong khóa học nói về câu hỏi này, nên mọi câu trả lời
        đều chỉ có thể đến từ kiến thức sẵn có của LLM."""
        state["response"] = REFUSAL_OUT_OF_CORPUS
        state["sources"] = []
        return state

    def _route_based_on_label(self, state: AgentState) -> Literal["chat", "learning", "refuse"]:
        """Route to appropriate handler based on the 4-way label.

        `cross_module` đi CHUNG nhánh với `learning`: vẫn sinh câu trả lời RAG, chỉ khác ở ngữ cảnh
        (`analyze()` đã ghép sẵn chunk của module chứa nội dung) và ở câu gợi ý module mà server nối
        vào cuối.
        """
        label = state.get("label")
        if label in ("learning", "cross_module"):
            return "learning"
        if label in _REFUSAL_LABELS:
            return "refuse"
        return "chat"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify", self._retrieve_and_classify)
        workflow.add_node("chat", self._handle_normal_chat)
        workflow.add_node("learning", self._handle_learning_query)
        workflow.add_node("refuse", self._handle_refusal)

        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_based_on_label,
            {
                "chat": "chat",
                "learning": "learning",
                "refuse": "refuse",
            }
        )
        workflow.add_edge("chat", END)
        workflow.add_edge("learning", END)
        workflow.add_edge("refuse", END)

        return workflow.compile()

    def process(
        self,
        query: str,
        course_id: Optional[str] = None,
        module_id: Optional[str] = None,
        history: Optional[list] = None
    ) -> dict:
        """Process user query through the agent (non-streaming)."""
        initial_state = {
            "query": query,
            "intent": None,
            "label": None,
            "suggested_module_id": None,
            "docs": None,
            "course_id": course_id,
            "module_id": module_id,
            "history": history if history else [],
            "response": None,
            "sources": None
        }

        final_state = self.graph.invoke(initial_state)

        label = final_state.get("label")
        refused = label in _REFUSAL_LABELS
        return {
            "response": final_state["response"],
            "intent": final_state["intent"],
            "label": label,
            "refused": refused,
            "refusal_reason": label if refused else None,
            "suggested_module_id": final_state.get("suggested_module_id"),
            "sources": final_state.get("sources", [])
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _build_chat_messages(self, query: str, history: list) -> list:
        """Build messages for a normal chat call."""
        messages = [{"role": "system", "content": _DEFAULT_CHAT_SYSTEM_PROMPT}]

        if history:
            recent_history = history[-15:] if len(history) > 15 else history
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages

    def process_stream(
        self,
        query: str,
        course_id: Optional[str] = None,
        module_id: Optional[str] = None,
        history: Optional[list] = None
    ) -> Generator[Dict, None, None]:
        """Process user query with streaming response.

        1. Truy xuất + phân loại 4 nhãn (dùng ĐÚNG `_retrieve_and_classify` mà graph node dùng,
           nên luồng stream và luồng non-stream không thể lệch nhau).
        2. Rẽ 3 nhánh: từ chối out_of_corpus (không gọi LLM) | learning + cross_module (RAG) | chat.

        Yields:
            {"type": "token", "delta": "..."} for text chunks.
            {"type": "done", "intent", "label", "refused", "refusal_reason",
             "suggested_module_id", "sources": [...]} as final event.
        """
        history = history or []

        state: AgentState = {
            "query": query,
            "intent": None,
            "label": None,
            "suggested_module_id": None,
            "docs": None,
            "course_id": course_id,
            "module_id": module_id,
            "history": history,
            "response": None,
            "sources": None,
        }
        state = self._retrieve_and_classify(state)
        label = state["label"]
        intent = state["intent"]
        refused = label in _REFUSAL_LABELS
        done_meta = {
            "type": "done",
            "intent": intent,
            "label": label,
            "refused": refused,
            "refusal_reason": label if refused else None,
            "suggested_module_id": state.get("suggested_module_id"),
        }

        if refused:
            # Từ chối phát như một event `token` bình thường (không phải type mới) — server/eval
            # đều cộng dồn text từ event `token`, và câu từ chối là hằng số, không gọi LLM.
            yield {"type": "token", "delta": REFUSAL_OUT_OF_CORPUS}
            yield {**done_meta, "sources": []}
            return

        if label in ("learning", "cross_module"):
            for event in self.rag_service.answer_question_stream(
                query=query,
                context_docs=state.get("docs") or [],
                history=history,
            ):
                if event["type"] == "token":
                    yield event
                elif event["type"] == "done":
                    yield {**done_meta, "sources": event.get("sources", [])}
        else:
            # Normal chat — stream directly from OpenAI
            messages = self._build_chat_messages(query, history)
            stream = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "delta": delta.content}

            yield {**done_meta, "sources": []}
