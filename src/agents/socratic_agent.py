
import os
from typing import TypedDict, Literal, Optional, List, Dict, Any

from openai import OpenAI
from langgraph.graph import StateGraph, END
from src.services.llm_client_factory import build_llm_client

from src.utils.prompt_loader import read_prompt


class AgentState(TypedDict):
    """State for the agent graph."""
    query: str                  # Câu hỏi của học sinh
    intent: Optional[str]       # Ý định: "chat" hoặc "learning"
    problem: Optional[str]      # Đề bài
    history: Optional[list]     # Lịch sử hội thoại
    response: Optional[str]     # Phản hồi của agent
    sources: Optional[list]     # Nguồn thông tin (nếu có)


INTENT_CLASSIFICATION_PROMPT = read_prompt(
    "socratic_agent", "classify_intent_prompt.txt",
    "Phân loại ý định của user thành learning hoặc chat. Chỉ trả về learning hoặc chat."
)

SYSTEM_PROMPT = read_prompt(
    "socratic_agent", "system_prompt.txt",
    "Bạn là Socratic tutor. Trả lời tiếng Việt ngắn gọn, dùng câu hỏi gợi mở."
)


class SocraticAgent:

    def __init__(self, llm_provider: str = "open_ai"):
        """Initialize the agent with LLM service."""
        self.llm_client, self.model = build_llm_client(llm_provider)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.graph = self._build_graph()


    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent: normal chat or learning-related."""
        query = state["query"]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0,
            max_tokens=10
        )

        raw_intent = (response.choices[0].message.content or "").strip().lower()
        if "out_of_domain" in raw_intent:
            normalized_intent = "out_of_domain"
        elif "learning" in raw_intent:
            normalized_intent = "learning"
        elif "chat" in raw_intent:
            normalized_intent = "chat"
        else:
            normalized_intent = "chat"

        next_state = dict(state)
        next_state["intent"] = normalized_intent
        return next_state
        
    def _ask_followup(self, state: AgentState) -> AgentState:
        """Ask follow-up questions to clarify learning intent."""
        query = state["query"]
        history = state.get("history", [])

        problem = state.get("problem")
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if problem:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Ngữ cảnh tài liệu/bài toán hiện tại. Hãy sử dụng ngữ cảnh này để phản hồi Socratic:\n"
                        f"{problem}"
                    )
                }
            )

        if history:
            recent_history = history[-12:] if len(history) > 12 else history
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        next_state = dict(state)
        next_state["response"] = (response.choices[0].message.content or "").strip()
        return next_state

    def _general_chat_response(self, state: AgentState) -> AgentState:
        """Handle non-learning conversation in a lightweight way."""
        query = state["query"]
        history = state.get("history", [])

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            recent_history = history[-8:] if len(history) > 8 else history
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        next_state = dict(state)
        next_state["response"] = (response.choices[0].message.content or "").strip()
        return next_state

    def _refuse_to_answer(self, state: AgentState) -> AgentState:
        """Refuse to answer when query is outside current problem scope."""
        next_state = dict(state)
        next_state["response"] = (
            "Thầy chỉ hỗ trợ các câu hỏi liên quan đến bài học hoặc đề bài hiện tại. "
            "Em hãy hỏi lại đúng nội dung bài này nhé."
        )
        return next_state

    def _route_by_intent(self, state: AgentState) -> Literal["learning", "chat", "refuse"]:
        """Route graph flow based on classified intent."""
        if state.get("intent") == "learning":
            return "learning"
        elif state.get("intent") == "chat":
            return "chat"
        else:
            return "refuse"

    def _build_graph(self):
        """Build and compile full LangGraph state machine for Socratic flow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("classify", self._classify_intent)
        workflow.add_node("learning", self._ask_followup)
        workflow.add_node("chat", self._general_chat_response)
        workflow.add_node("refuse", self._refuse_to_answer)

        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_by_intent,
            {
                "learning": "learning",
                "chat": "chat",
                "refuse": "refuse"
            }
        )
        workflow.add_edge("learning", END)
        workflow.add_edge("chat", END)
        workflow.add_edge("refuse", END)

        return workflow.compile()

    def process(
        self,
        query: str,
        problem: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Process one Socratic turn with optional preloaded problem context."""
        state: AgentState = {
            "query": query,
            "intent": None,
            "problem": problem,
            "history": history or [],
            "response": None,
            "sources": []
        }

        final_state = self.graph.invoke(state)

        return {
            "response": final_state.get("response") or "",
            "intent": final_state.get("intent") or "chat",
            "sources": final_state.get("sources") or []
        }
