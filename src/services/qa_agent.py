"""QA Oral Exam Agent - Conducts an interactive oral examination for code understanding.

Behaviour contract (per turn):
─────────────────────────────────────────────────────────────────────────────
Turn type         | current_question_turns | Action
─────────────────────────────────────────────────────────────────────────────
Greeting / start  | N/A (start_session)    | Introduce exam + pose Q1
First answer      | 0                      | Evaluate → lock score → give natural
                  |                        |   feedback + optional follow-up probe
Follow-up #1      | 1                      | Continue discussion; advance if natural
Max follow-up #2  | ≥ 2                    | Wrap up topic; transition to next Q
                  |                        |   (set advance_to_next=True)
Last question done| is_last_question=True  | Close session (is_complete=True)
─────────────────────────────────────────────────────────────────────────────

Grading rules:
  - Score is determined ONLY on the first answer (turns == 0).
  - Score 2 : answer demonstrates clear understanding + correct concepts.
  - Score 1 : partially correct or incomplete.
  - Score 0 : incorrect or completely off-topic.
  - Subsequent turns (follow-ups) do NOT change the locked score.
"""
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI
from src.services.llm_client_factory import build_llm_client

logger = logging.getLogger(__name__)

# Max follow-up turns per question before forcing transition
MAX_FOLLOW_UPS = 2

_TEACHER_PERSONA = """Bạn là giảng viên đang thực hiện buổi vấn đáp (oral exam) về bài tập lập trình.

Phong cách:
- Thân thiện nhưng chuyên nghiệp, như giảng viên thực tế
- Tiếng Việt tự nhiên (có thể dùng thuật ngữ kỹ thuật bằng tiếng Anh)
- Không kể điểm số ra bên ngoài ("tôi cho em X điểm") — đánh giá ngầm qua lời phản hồi
- Đặt câu hỏi gợi mở, không tiết lộ đáp án ngay
- Có thể hỏi ngược lại nếu sinh viên hỏi bạn (nhưng khuyến khích họ tự suy nghĩ trước)"""


class QAAgentService:
    """Interactive oral exam agent for code understanding assessment."""

    def __init__(self, llm_provider: str = "open_ai") -> None:
        self.llm_client, self.model = build_llm_client(llm_provider)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # ------------------------------------------------------------------
    # Public: start_session
    # ------------------------------------------------------------------

    def start_session(
        self,
        question_bank: List[Dict[str, Any]],
        project_summary: str,
        max_questions: int = 5,
    ) -> str:
        """Return the opening greeting and the first question.

        Args:
            question_bank:   Pre-generated question list (from QuestionGeneratorService).
            project_summary: Short plain-text project summary.
            max_questions:   Maximum number of questions to ask. Bank is truncated to this.

        Returns:
            Opening response string (no scoring at this point).
        """
        # Enforce the question limit
        question_bank = question_bank[:max_questions]

        if not question_bank:
            return (
                "Chào em! Thầy/cô đã xem qua bài làm của em. "
                "Em có muốn chia sẻ về những gì em đã làm trong bài không?"
            )

        first_q = question_bank[0]["question"]
        total_q = len(question_bank)

        system = _TEACHER_PERSONA
        user_prompt = (
            f"=== TÓM TẮT PROJECT ===\n{project_summary}\n\n"
            f"=== CÂU HỎI ĐẦU TIÊN (1/{total_q}) ===\n{first_q}\n\n"
            "Hãy viết lời chào mừng và đặt câu hỏi đầu tiên. "
            "Đề cập ngắn gọn rằng đây là buổi vấn đáp để hiểu thêm về bài làm của em, "
            "không phải để thêm điểm phạt. Kết thúc bằng câu hỏi đầu tiên."
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("start_session LLM call failed: %s", exc)
            return (
                f"Chào em! Thầy/cô đã xem qua bài làm của em rồi. "
                f"Hôm nay chúng ta sẽ có buổi trao đổi ngắn về project. "
                f"Câu hỏi đầu tiên: {first_q}"
            )

    # ------------------------------------------------------------------
    # Public: process_answer
    # ------------------------------------------------------------------

    def process_answer(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        current_question_turns: int,
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool = False,
    ) -> Dict[str, Any]:
        """Process one student turn and return the teacher's response + metadata.

        Args:
            user_message:           Student's message for this turn.
            current_question:       QuestionBankItem dict being discussed.
            current_question_turns: 0 = first answer; 1+ = follow-up turns.
            score_locked:           Already-locked score dict or None.
            project_summary:        Short project description for context.
            history:                Recent conversation messages (no big JSON blobs).
            is_last_question:       Whether this is the final question in the bank.

        Returns:
            {
                "response":       str,
                "scoring_event":  {"question_id", "score", "max_score"} | None,
                "advance_to_next": bool,
                "is_complete":    bool,
            }
        """
        if current_question_turns == 0:
            return self._handle_first_answer(
                user_message, current_question, project_summary, history
            )

        if current_question_turns >= MAX_FOLLOW_UPS:
            return self._handle_max_followup_reached(
                user_message, current_question, score_locked,
                project_summary, history, is_last_question
            )

        # Follow-up turn (turns == 1, or < MAX_FOLLOW_UPS)
        return self._handle_followup(
            user_message, current_question, score_locked,
            project_summary, history, is_last_question
        )

    # ------------------------------------------------------------------
    # Private: first answer — evaluate + lock score
    # ------------------------------------------------------------------

    def _handle_first_answer(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        project_summary: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rubric = current_question.get("scoring_rubric", {})
        expected = current_question.get("expected_concepts", [])

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CÂU HỎI ĐANG HỎI ===\n{current_question['question']}\n\n"
            f"=== CONCEPTS KỲ VỌNG ===\n{', '.join(expected)}\n\n"
            f"=== RUBRIC CHẤM ĐIỂM (nội bộ, không tiết lộ) ===\n"
            f"2 điểm: {rubric.get('score_2', '')}\n"
            f"1 điểm: {rubric.get('score_1', '')}\n"
            f"0 điểm: {rubric.get('score_0', '')}\n\n"
            "Đây là câu trả lời ĐẦU TIÊN của sinh viên cho câu hỏi này.\n"
            "Nhiệm vụ của bạn:\n"
            "1. Đánh giá câu trả lời theo rubric. Chọn điểm: 0, 1, hoặc 2.\n"
            "2. Phản hồi tự nhiên bằng tiếng Việt như giảng viên vấn đáp:\n"
            "   - Điểm 2: Khen ngợi, xác nhận những gì đúng, có thể đặt câu hỏi chuyên sâu hơn\n"
            "   - Điểm 1: Ghi nhận những gì đúng, khéo léo chỉ ra phần còn thiếu, hỏi thêm để hiểu hơn\n"
            "   - Điểm 0: Khuyến khích, gợi ý hướng suy nghĩ (không đưa đáp án thẳng), hỏi lại theo cách khác\n"
            "3. KHÔNG nói điểm số cụ thể trong phản hồi ('tôi cho em X điểm').\n"
            "4. Cuối phản hồi có thể đặt câu hỏi tiếp theo ngắn để duy trì cuộc trò chuyện.\n\n"
            "Trả về JSON:\n"
            '{"score": 0|1|2, "response": "..."}'
        )

        messages = self._build_messages(system, history, user_message)

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            score = int(result.get("score", 0))
            response_text = result.get("response", "")
        except Exception as exc:
            logger.error("First-answer evaluation failed: %s", exc, exc_info=True)
            score = 0
            response_text = (
                "Cảm ơn câu trả lời của em. Thầy/cô sẽ tiếp tục với câu hỏi tiếp theo nhé."
            )

        # Clamp score to valid values
        if score not in (0, 1, 2):
            score = 0 if score < 1 else (1 if score < 2 else 2)

        return {
            "response": response_text,
            "scoring_event": {
                "question_id": current_question["id"],
                "score": score,
                "max_score": current_question.get("max_score", 2),
            },
            "advance_to_next": False,
            "is_complete": False,
        }

    # ------------------------------------------------------------------
    # Private: follow-up turn (turns == 1, grade already locked)
    # ------------------------------------------------------------------

    def _handle_followup(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool,
    ) -> Dict[str, Any]:
        """Handle the first follow-up exchange after grading."""
        locked_score = (score_locked or {}).get("score", 0)

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CHỦ ĐỀ ĐANG THẢO LUẬN ===\n{current_question['question']}\n\n"
            f"=== TRẠNG THÁI ===\n"
            f"Sinh viên đã trả lời câu hỏi này (điểm đã được ghi nhận nội bộ). "
            f"Đây là lượt trao đổi thêm về cùng chủ đề.\n\n"
            "Nhiệm vụ:\n"
            "- Tiếp tục cuộc trò chuyện tự nhiên\n"
            + (
                "- Điểm thấp: giải thích nhẹ nhàng khái niệm đúng, giúp sinh viên hiểu (không phán xét)\n"
                if locked_score < 2
                else "- Điểm cao: khám phá thêm, hỏi chuyên sâu hơn\n"
            )
            + "- Nếu cuộc trò chuyện về chủ đề này đã đủ, kết thúc topic và báo hiệu "
              "chuyển sang câu hỏi tiếp (nói ví dụ: 'Vậy thầy/cô sẽ hỏi thêm một câu khác...').\n"
            "- Nếu chuyển sang câu hỏi tiếp, thêm field advance=true vào JSON.\n"
            + (
                "- ĐÂY LÀ CÂU HỎI CUỐI CÙNG. Nếu advance=true, thay vì dẫn sang câu tiếp, "
                "hãy kết thúc buổi vấn đáp với: tóm tắt ngắn, lời chúc mừng sinh viên đã hoàn thành, "
                "lời động viên tích cực. KHÔNG đặt thêm câu hỏi mới.\n"
                if is_last_question
                else ""
            )
            + "\nTrả về JSON:\n"
            '{"response": "...", "advance": false}'
        )

        messages = self._build_messages(system, history, user_message)

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            advance = bool(result.get("advance", False))
            response_text = result.get("response", "")
        except Exception as exc:
            logger.error("Follow-up handler failed: %s", exc, exc_info=True)
            advance = False
            response_text = "Em có muốn hỏi thêm gì về chủ đề này không?"

        is_complete = advance and is_last_question

        return {
            "response": response_text,
            "scoring_event": None,
            "advance_to_next": advance,
            "is_complete": is_complete,
        }

    # ------------------------------------------------------------------
    # Private: max follow-ups reached — force transition
    # ------------------------------------------------------------------

    def _handle_max_followup_reached(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool,
    ) -> Dict[str, Any]:
        """After 2 follow-up turns, transition to the next question (or close session)."""
        closing_instruction = (
            "Đây là câu hỏi CUỐI CÙNG trong buổi vấn đáp. Khi kết thúc câu hỏi này, hãy:\n"
            "1. Tổng kết ngắn gọn toàn bộ buổi vấn đáp.\n"
            "2. Chúc mừng sinh viên đã hoàn thành xuất sắc buổi vấn đáp.\n"
            "3. Gửi lời động viên tích cực để sinh viên tự tin hơn.\n"
            "KHÔNG đặt thêm câu hỏi mới."
            if is_last_question
            else "Hãy khéo léo kết thúc chủ đề này và thông báo chuyển sang câu hỏi tiếp theo."
        )

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CHỦ ĐỀ VỪA THẢO LUẬN ===\n{current_question['question']}\n\n"
            f"{closing_instruction}"
        )

        messages = self._build_messages(system, history, user_message)

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=400,
            )
            response_text = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Max-followup handler failed: %s", exc, exc_info=True)
            response_text = (
                "Cảm ơn em đã chia sẻ. Thầy/cô sẽ chuyển sang câu hỏi tiếp theo nhé."
                if not is_last_question
                else "Cảm ơn em rất nhiều! Buổi vấn đáp kết thúc tại đây."
            )

        return {
            "response": response_text,
            "scoring_event": None,
            "advance_to_next": True,
            "is_complete": is_last_question,
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        system_prompt: str,
        history: List[Dict[str, Any]],
        current_user_message: str,
    ) -> List[Dict[str, str]]:
        """Build the messages list for an LLM call.

        Only user/assistant messages from history are included.
        System messages with large JSON blobs are intentionally excluded.
        """
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Keep only user/assistant from history (last 10 messages to stay concise)
        recent = [
            m for m in history
            if m.get("role") in ("user", "assistant")
        ][-10:]

        for m in recent:
            content = m.get("content", "")
            # Safety: skip anything that looks like a huge JSON repo dump
            if isinstance(content, dict) or (isinstance(content, str) and len(content) > 3000):
                continue
            msgs.append({"role": m["role"], "content": str(content)})

        msgs.append({"role": "user", "content": current_user_message})
        return msgs

    # ==================================================================
    # Streaming variants
    # ==================================================================

    def start_session_stream(
        self,
        question_bank: List[Dict[str, Any]],
        project_summary: str,
        max_questions: int = 5,
    ) -> Generator[Dict, None, None]:
        """Streaming version of start_session. Yields token events + done."""
        # Enforce the question limit
        question_bank = question_bank[:max_questions]

        if not question_bank:
            fallback = (
                "Chào em! Thầy/cô đã xem qua bài làm của em. "
                "Em có muốn chia sẻ về những gì em đã làm trong bài không?"
            )
            yield {"type": "token", "delta": fallback}
            yield {"type": "done"}
            return

        first_q = question_bank[0]["question"]
        total_q = len(question_bank)

        system = _TEACHER_PERSONA
        user_prompt = (
            f"=== TÓM TẮT PROJECT ===\n{project_summary}\n\n"
            f"=== CÂU HỎI ĐẦU TIÊN (1/{total_q}) ===\n{first_q}\n\n"
            "Hãy viết lời chào mừng và đặt câu hỏi đầu tiên. "
            "Đề cập ngắn gọn rằng đây là buổi vấn đáp để hiểu thêm về bài làm của em, "
            "không phải để thêm điểm phạt. Kết thúc bằng câu hỏi đầu tiên."
        )

        try:
            stream = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "delta": delta.content}
        except Exception as exc:
            logger.error("start_session_stream LLM call failed: %s", exc)
            yield {
                "type": "token",
                "delta": f"Chào em! Thầy/cô đã xem qua bài làm của em rồi. "
                         f"Hôm nay chúng ta sẽ có buổi trao đổi ngắn về project. "
                         f"Câu hỏi đầu tiên: {first_q}",
            }

        yield {"type": "done"}

    def process_answer_stream(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        current_question_turns: int,
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool = False,
    ) -> Generator[Dict, None, None]:
        """Streaming version of process_answer."""
        if current_question_turns == 0:
            yield from self._handle_first_answer_stream(
                user_message, current_question, project_summary, history
            )
        elif current_question_turns >= MAX_FOLLOW_UPS:
            yield from self._handle_max_followup_stream(
                user_message, current_question, score_locked,
                project_summary, history, is_last_question
            )
        else:
            yield from self._handle_followup_stream(
                user_message, current_question, score_locked,
                project_summary, history, is_last_question
            )

    # -- First answer (stream) ------------------------------------------

    def _handle_first_answer_stream(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        project_summary: str,
        history: List[Dict[str, Any]],
    ) -> Generator[Dict, None, None]:
        """Collect full JSON (for scoring), then emit text + metadata.

        JSON mode with stream=True returns JSON tokens — we accumulate them,
        parse the result, then yield the response text and scoring event.
        """
        rubric = current_question.get("scoring_rubric", {})
        expected = current_question.get("expected_concepts", [])

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CÂU HỎI ĐANG HỎI ===\n{current_question['question']}\n\n"
            f"=== CONCEPTS KỲ VỌNG ===\n{', '.join(expected)}\n\n"
            f"=== RUBRIC CHẤM ĐIỂM (nội bộ, không tiết lộ) ===\n"
            f"2 điểm: {rubric.get('score_2', '')}\n"
            f"1 điểm: {rubric.get('score_1', '')}\n"
            f"0 điểm: {rubric.get('score_0', '')}\n\n"
            "Đây là câu trả lời ĐẦU TIÊN của sinh viên cho câu hỏi này.\n"
            "Nhiệm vụ của bạn:\n"
            "1. Đánh giá câu trả lời theo rubric. Chọn điểm: 0, 1, hoặc 2.\n"
            "2. Phản hồi tự nhiên bằng tiếng Việt như giảng viên vấn đáp.\n"
            "3. KHÔNG nói điểm số cụ thể trong phản hồi.\n\n"
            "Trả về JSON:\n"
            '{"score": 0|1|2, "response": "..."}'
        )

        messages = self._build_messages(system, history, user_message)

        score = 0
        response_text = ""

        try:
            # Collect full JSON with streaming to keep connection alive
            collected = ""
            stream = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=600,
                response_format={"type": "json_object"},
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected += delta.content

            result = json.loads(collected)
            score = int(result.get("score", 0))
            response_text = result.get("response", "")
        except Exception as exc:
            logger.error("First-answer stream failed: %s", exc, exc_info=True)
            score = 0
            response_text = "Cảm ơn câu trả lời của em. Thầy/cô sẽ tiếp tục với câu hỏi tiếp theo nhé."

        if score not in (0, 1, 2):
            score = 0 if score < 1 else (1 if score < 2 else 2)

        # Emit the response text as a single token event
        yield {"type": "token", "delta": response_text}
        yield {
            "type": "done",
            "scoring_event": {
                "question_id": current_question["id"],
                "score": score,
                "max_score": current_question.get("max_score", 2),
            },
            "advance_to_next": False,
            "is_complete": False,
        }

    # -- Follow-up (stream) ---------------------------------------------

    def _handle_followup_stream(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool,
    ) -> Generator[Dict, None, None]:
        """Collect JSON (advance flag), then emit text + metadata."""
        locked_score = (score_locked or {}).get("score", 0)

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CHỦ ĐỀ ĐANG THẢO LUẬN ===\n{current_question['question']}\n\n"
            f"=== TRẠNG THÁI ===\n"
            f"Sinh viên đã trả lời câu hỏi này (điểm đã được ghi nhận nội bộ). "
            f"Đây là lượt trao đổi thêm về cùng chủ đề.\n\n"
            "Nhiệm vụ:\n"
            "- Tiếp tục cuộc trò chuyện tự nhiên\n"
            + (
                "- Điểm thấp: giải thích nhẹ nhàng khái niệm đúng, giúp sinh viên hiểu\n"
                if locked_score < 2
                else "- Điểm cao: khám phá thêm, hỏi chuyên sâu hơn\n"
            )
            + "- Nếu cuộc trò chuyện về chủ đề này đã đủ, kết thúc topic này.\n"
            "- Nếu chuyển sang câu hỏi tiếp, thêm field advance=true vào JSON.\n"
            + (
                "- ĐÂY LÀ CÂU HỎI CUỐI CÙNG. Nếu advance=true, thay vì dẫn sang câu tiếp, "
                "hãy kết thúc buổi vấn đáp với: tóm tắt ngắn, lời chúc mừng sinh viên đã hoàn thành, "
                "lời động viên tích cực. KHÔNG đặt thêm câu hỏi mới.\n"
                if is_last_question
                else ""
            )
            + "\nTrả về JSON:\n"
            '{"response": "...", "advance": false}'
        )

        messages = self._build_messages(system, history, user_message)
        advance = False
        response_text = ""

        try:
            collected = ""
            stream = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected += delta.content

            result = json.loads(collected)
            advance = bool(result.get("advance", False))
            response_text = result.get("response", "")
        except Exception as exc:
            logger.error("Follow-up stream failed: %s", exc, exc_info=True)
            response_text = "Em có muốn hỏi thêm gì về chủ đề này không?"

        is_complete = advance and is_last_question

        yield {"type": "token", "delta": response_text}
        yield {
            "type": "done",
            "scoring_event": None,
            "advance_to_next": advance,
            "is_complete": is_complete,
        }

    # -- Max follow-up reached (stream) ---------------------------------

    def _handle_max_followup_stream(
        self,
        user_message: str,
        current_question: Dict[str, Any],
        score_locked: Optional[Dict[str, Any]],
        project_summary: str,
        history: List[Dict[str, Any]],
        is_last_question: bool,
    ) -> Generator[Dict, None, None]:
        """Stream text directly (no JSON parsing needed)."""
        closing_instruction = (
            "Đây là câu hỏi CUỐI CÙNG trong buổi vấn đáp. Khi kết thúc câu hỏi này, hãy:\n"
            "1. Tổng kết ngắn gọn toàn bộ buổi vấn đáp.\n"
            "2. Chúc mừng sinh viên đã hoàn thành xuất sắc buổi vấn đáp.\n"
            "3. Gửi lời động viên tích cực để sinh viên tự tin hơn.\n"
            "KHÔNG đặt thêm câu hỏi mới."
            if is_last_question
            else "Hãy khéo léo kết thúc chủ đề này và thông báo chuyển sang câu hỏi tiếp theo."
        )

        system = (
            f"{_TEACHER_PERSONA}\n\n"
            f"=== CONTEXT PROJECT ===\n{project_summary}\n\n"
            f"=== CHỦ ĐỀ VỪA THẢO LUẬN ===\n{current_question['question']}\n\n"
            f"{closing_instruction}"
        )

        messages = self._build_messages(system, history, user_message)

        try:
            stream = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=400,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "delta": delta.content}
        except Exception as exc:
            logger.error("Max-followup stream failed: %s", exc, exc_info=True)
            fallback = (
                "Cảm ơn em đã chia sẻ. Thầy/cô sẽ chuyển sang câu hỏi tiếp theo nhé."
                if not is_last_question
                else "Cảm ơn em rất nhiều! Buổi vấn đáp kết thúc tại đây."
            )
            yield {"type": "token", "delta": fallback}

        yield {
            "type": "done",
            "scoring_event": None,
            "advance_to_next": True,
            "is_complete": is_last_question,
        }
