"""Project QA pipeline agent built with LangGraph.

This module provides:
- QuestionGeneratorAgent
- InterviewAgent (interrupt/resume with MemorySaver)
- GradingAgent
- ProjectQAOrchestrator
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Literal, Optional
from uuid import uuid4
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from openai import OpenAI
from src.services.llm_client_factory import build_llm_client

logger = logging.getLogger(__name__)
QUESTION_COUNT = 3
PER_QUESTION_MAX = 2  # Thang điểm mỗi câu: 0 / 1 / 2 (thống nhất với grading_system_prompt "thang 0-2")

# Ngưỡng điểm hợp lệ của bài nộp (0-100). Dưới ngưỡng này bài nộp bị coi là không đạt yêu cầu đề bài.
SUBMISSION_VALIDITY_THRESHOLD = 40

# Ba trạng thái xác thực bài nộp:
# - "valid"   : bài nộp bám đúng đề bài
# - "invalid" : bài nộp sai đề / nộp nhầm repo / không giải quyết yêu cầu → chấm 0
# - "unknown" : KHÔNG kết luận được (không có bài nộp, không có đề bài đáng tin, hoặc lỗi hệ thống).
#               Trạng thái này KHÔNG được phép ép 0 điểm — im lặng cho qua thì còn tệ hơn là nói thẳng
#               rằng hệ thống chưa đối chiếu được.
ValidationStatus = Literal["valid", "invalid", "unknown"]

VERDICT_LABELS: dict[str, str] = {
    "valid": "Bài làm đạt yêu cầu",
    "invalid": "Bài làm không đạt yêu cầu",
    "unknown": "Chưa xác định được mức độ phù hợp với đề bài",
}


# ---------------------------------------------------------------------------
# Shared Types
# ---------------------------------------------------------------------------

class Question(TypedDict):
    question_id: str
    content: str
    intent: str
    rubric_ref: Optional[str]
    difficulty: Literal["basic", "intermediate", "advanced"]


class StudentSubmission(TypedDict):
    code: Optional[str]
    text: Optional[str]
    language: Optional[str]
    notes: Optional[str]


class SubmissionValidationResult(TypedDict):
    is_valid: bool
    validity_score: int
    status: ValidationStatus
    verdict_label: str
    issues: list[str]
    summary: str


class ChatTurn(TypedDict):
    role: Literal["agent", "student"]
    content: str
    turn_index: int
    question_id: str
    timestamp: str


class QuestionScore(TypedDict):
    question_id: str
    question_content: str
    score: Literal[0, 1, 2]
    score_label: Literal["không hiểu", "hiểu một phần", "hiểu hoàn toàn"]
    what_student_got_right: str
    what_student_missed: str
    key_evidence: str
    rubric_criteria_met: Optional[str]


class GradingReport(TypedDict):
    student_name: Optional[str]
    total_score: int
    max_score: int
    percentage: float
    score_out_of_10: float
    summary: str
    per_question: list[QuestionScore]
    generated_at: str

    # Kết quả xác thực bài nộp (tính ở Agent 1) được đính kèm báo cáo để học sinh nhìn thấy
    # vì sao mình bị 0 điểm, thay vì chỉ thấy điểm 0 mà không có lý do.
    submission_status: ValidationStatus
    submission_verdict: str
    submission_valid: bool
    submission_validity_score: int
    submission_issues: list[str]
    submission_summary: str
    forced_zero: bool


class PipelineState(TypedDict):
    project_description: str
    rubric: Optional[str]
    student_name: Optional[str]
    student_submission: Optional[StudentSubmission]
    submission_validation: Optional[SubmissionValidationResult]
    questions: Optional[list[Question]]
    full_chat_log: Optional[list[ChatTurn]]
    grading_report: Optional[GradingReport]
    pipeline_stage: Literal["questioning", "interview", "grading", "done"]
    pipeline_error: Optional[str]


# ---------------------------------------------------------------------------
# Question Generator Agent
# ---------------------------------------------------------------------------

class QuestionGeneratorState(TypedDict):
    project_description: str
    rubric: Optional[str]
    student_submission: Optional[StudentSubmission]
    raw_llm_output: str
    questions: list[Question]
    generation_status: Literal["pending", "success", "failed"]
    error_message: Optional[str]
    retry_count: int
    input_valid: bool

    # Bật/tắt bước đối chiếu bài nộp với đề bài. Caller phải tắt khi project_description
    # không phải đề bài thật (ví dụ nó được tóm tắt ra TỪ CHÍNH bài nộp) — lúc đó việc
    # đối chiếu là so bài nộp với mô tả của chính nó, kết quả luôn "hợp lệ" và vô nghĩa.
    validate_submission: bool

    submission_valid: bool
    submission_validity_score: int
    submission_status: ValidationStatus
    submission_issues: list[str]
    submission_summary: str


class QuestionGeneratorAgent:
    """Generate interview questions from project context."""

    def __init__(self, llm_client: OpenAI, model: str) -> None:
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = _read_prompt_file(
            "question_generator_system_prompt.txt",
            "Ban la QuestionGeneratorAgent. Tra ve JSON array gom dung 3 cau hoi."
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(QuestionGeneratorState)
        workflow.add_node("parse_input", self._parse_input)
        workflow.add_node("validate_submission", self._validate_submission)
        workflow.add_node("generate_questions_llm", self._generate_questions_llm)
        workflow.add_node("validate_questions", self._validate_questions)
        workflow.add_node("regenerate_questions", self._regenerate_questions)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "validate_submission")
        workflow.add_edge("validate_submission", "generate_questions_llm")
        workflow.add_edge("generate_questions_llm", "validate_questions")
        workflow.add_conditional_edges(
            "validate_questions",
            self._route_after_validation,
            {"regenerate": "regenerate_questions", "finalize": "finalize"},
        )
        workflow.add_edge("regenerate_questions", "generate_questions_llm")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _parse_input(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        try:
            project_description = (state.get("project_description") or "").strip()
            rubric = state.get("rubric")
            next_state = dict(state)
            next_state["project_description"] = project_description
            next_state["rubric"] = rubric.strip() if isinstance(rubric, str) else rubric
            next_state["generation_status"] = "pending"
            next_state["error_message"] = None
            next_state["input_valid"] = bool(project_description)
            next_state.setdefault("retry_count", 0)
            next_state.setdefault("raw_llm_output", "")
            next_state.setdefault("questions", [])
            next_state.setdefault("student_submission", None)
            next_state.setdefault("validate_submission", True)
            next_state.setdefault("submission_valid", True)
            next_state.setdefault("submission_validity_score", 0)
            next_state.setdefault("submission_status", "unknown")
            next_state.setdefault("submission_issues", [])
            next_state.setdefault("submission_summary", "")
            if not project_description:
                next_state["generation_status"] = "failed"
                next_state["error_message"] = "project_description is required"
            return next_state
        except Exception as exc:
            logger.error("parse_input failed: %s", exc, exc_info=True)
            next_state = dict(state)
            next_state["generation_status"] = "failed"
            next_state["error_message"] = str(exc)
            return next_state

    def _validate_submission(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        """Đối chiếu bài nộp với đề bài, cho ra điểm phù hợp 0-100.

        Trả về ba trạng thái chứ không phải hai. Trước đây mọi trường hợp "không kết luận
        được" (không có bài nộp, LLM lỗi, JSON hỏng) đều bị quy về valid/100, nên một repo
        hoàn toàn lạc đề vẫn lọt qua mà không ai biết. Giờ những trường hợp đó là "unknown":
        vẫn cho vấn đáp, không ép 0 điểm, nhưng nói rõ ra là chưa đối chiếu được.
        """
        next_state = dict(state)
        submission = state.get("student_submission")
        code_or_text = _get_submission_content(submission)

        if not code_or_text:
            return _apply_validation(
                next_state,
                status="unknown",
                score=0,
                issues=[],
                summary="Không có bài nộp để đối chiếu với đề bài.",
            )

        # project_description được sinh ra từ chính bài nộp → so sánh là vô nghĩa, bỏ qua.
        if not state.get("validate_submission", True):
            return _apply_validation(
                next_state,
                status="unknown",
                score=0,
                issues=[],
                summary=(
                    "Hệ thống chưa lấy được đề bài gốc nên không đối chiếu được bài nộp "
                    "với yêu cầu đề bài."
                ),
            )

        try:
            project_description = state.get("project_description", "")
            language = submission.get("language") if submission else None
            notes = submission.get("notes") if submission else None

            prompt = (
                "Đối chiếu bài nộp của học sinh với ĐỀ BÀI dưới đây và chấm mức độ phù hợp.\n\n"
                f"ĐỀ BÀI:\n{project_description}\n\n"
                f"NGÔN NGỮ/NGỮ CẢNH: {language or 'không rõ'}\n"
                f"GHI CHÚ CỦA HỌC SINH: {notes or 'không có'}\n\n"
                f"BÀI NỘP CỦA HỌC SINH:\n{code_or_text}\n\n"
                "CÁCH CHẤM validity_score (0-100) — bám sát các mốc sau:\n"
                "- 0-15: Bài nộp KHÔNG LIÊN QUAN đến đề bài (nộp nhầm repo, nhầm môn, "
                "code giải một bài toán hoàn toàn khác, hoặc chỉ là template/boilerplate trống).\n"
                "- 16-39: Có chạm tới chủ đề nhưng KHÔNG giải quyết yêu cầu chính của đề bài, "
                "hoặc thiếu phần lớn các chức năng bắt buộc.\n"
                "- 40-69: Giải quyết được yêu cầu chính nhưng còn thiếu sót hoặc có lỗi logic.\n"
                "- 70-100: Bám sát đề bài và giải quyết đúng các yêu cầu.\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. Chỉ so bài nộp với ĐỀ BÀI ở trên. Code viết đẹp, đầy đủ, chuyên nghiệp "
                "NHƯNG lạc đề thì vẫn phải chấm 0-15.\n"
                "2. Nếu không tìm được bằng chứng nào trong bài nộp tương ứng với yêu cầu "
                "của đề bài, chấm dưới 20.\n"
                "3. Đọc logic để đánh giá, không cần chạy thử.\n"
                "4. issues phải nêu cụ thể yêu cầu nào của đề bài không thấy trong bài nộp.\n"
                "5. summary viết 1-3 câu tiếng Việt có dấu, nói thẳng bài nộp có đúng đề hay không.\n\n"
                "Trả về JSON object đúng schema:\n"
                "{\n"
                '  "validity_score": 0-100,\n'
                '  "issues": ["..."],\n'
                '  "summary": "..."\n'
                "}"
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là giáo viên review bài nộp, nghiêm khắc về chuyện đúng đề. "
                            "Chỉ trả về JSON object hợp lệ."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )

            payload = _extract_json_object((response.choices[0].message.content or "").strip())

            if "validity_score" not in payload:
                raise ValueError("LLM response missing validity_score")

            validity_score = min(max(int(payload["validity_score"]), 0), 100)

            raw_issues = payload.get("issues")
            issues = (
                [str(item).strip() for item in raw_issues if str(item).strip()]
                if isinstance(raw_issues, list)
                else []
            )

            # Điểm là nguồn sự thật duy nhất. Trước đây còn phụ thuộc thêm cờ is_valid do LLM
            # tự khai, mà cờ đó mặc định True khi thiếu → bài lạc đề vẫn được coi là hợp lệ.
            status: ValidationStatus = (
                "valid" if validity_score >= SUBMISSION_VALIDITY_THRESHOLD else "invalid"
            )

            return _apply_validation(
                next_state,
                status=status,
                score=validity_score,
                issues=issues,
                summary=str(payload.get("summary") or "").strip(),
            )
        except Exception as exc:
            logger.warning("validate_submission failed, mark submission as unknown: %s", exc)
            return _apply_validation(
                next_state,
                status="unknown",
                score=0,
                issues=[],
                summary="Không thể xác thực bài nộp do lỗi hệ thống.",
            )

    def _generate_questions_llm(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        next_state = dict(state)
        try:
            if not state.get("input_valid"):
                return next_state

            project_description = state["project_description"]
            rubric = state.get("rubric")
            submission_context = _build_submission_context_for_prompt(state.get("student_submission"))
            submission_status: ValidationStatus = state.get("submission_status", "unknown")
            submission_summary = state.get("submission_summary") or "Khong co nhan xet submission."
            validity_score = int(state.get("submission_validity_score", 0))

            if submission_status == "invalid":
                # Bài nộp lạc đề: hỏi về bài nộp sẽ ra toàn câu hỏi lạc đề theo. Neo câu hỏi
                # vào đề bài để buổi vấn đáp vẫn kiểm tra đúng thứ cần kiểm tra.
                validation_instructions = (
                    f"CẢNH BÁO: bài nộp bị đánh giá KHÔNG ĐẠT yêu cầu đề bài "
                    f"(điểm phù hợp {validity_score}/100). "
                    "Hãy bám vào ĐỀ BÀI để ra câu hỏi, KHÔNG lấy nội dung bài nộp làm chuẩn. "
                    "Ít nhất một câu phải yêu cầu học sinh chỉ ra phần nào trong bài nộp "
                    "đáp ứng yêu cầu của đề bài."
                )
            else:
                validation_instructions = (
                    "Có thể dựa vào bài nộp để ra câu hỏi cụ thể, dẫn chiếu tới code của học sinh."
                )

            if rubric:
                rubric_instructions = (
                    "Rubric da duoc cung cap. Hay tao cau hoi bam sat rubric, "
                    "moi cau phai map voi mot tieu chi va dien vao rubric_ref."
                )
            else:
                rubric_instructions = (
                    "Khong co rubric. Hay suy luan tu mo ta project va phan bo do kho "
                    "deu giua basic/intermediate/advanced."
                )

            user_prompt = (
                f"PROJECT DESCRIPTION:\n{project_description}\n\n"
                f"RUBRIC:\n{rubric or 'null'}\n\n"
                f"STUDENT SUBMISSION CONTEXT:\n{submission_context}\n\n"
                f"SUBMISSION VALIDATION SUMMARY:\n{submission_summary}\n\n"
                f"INSTRUCTIONS:\n{rubric_instructions}\n{validation_instructions}\n"
                f"Tra ve JSON array dung {QUESTION_COUNT} phan tu theo schema Question. "
                f"question_id phai la q1..q{QUESTION_COUNT}."
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
            )

            content = (response.choices[0].message.content or "").strip()
            next_state["raw_llm_output"] = content
            return next_state
        except Exception as exc:
            logger.error("generate_questions_llm failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _validate_questions(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        next_state = dict(state)
        try:
            raw_output = state.get("raw_llm_output") or ""
            questions = _extract_json_array(raw_output)

            validated: list[Question] = []
            if not isinstance(questions, list):
                raise ValueError("LLM output is not a JSON array")
            if len(questions) != QUESTION_COUNT:
                raise ValueError(f"Question count must be exactly {QUESTION_COUNT}")

            for i, item in enumerate(questions, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Question #{i} is not an object")

                question_id = str(item.get("question_id") or f"q{i}").strip()
                content = str(item.get("content") or "").strip()
                intent = str(item.get("intent") or "").strip()
                rubric_ref = item.get("rubric_ref")
                difficulty = str(item.get("difficulty") or "").strip().lower()

                if question_id != f"q{i}":
                    question_id = f"q{i}"
                if not content or not intent:
                    raise ValueError(f"Question #{i} has empty content/intent")
                if difficulty not in {"basic", "intermediate", "advanced"}:
                    raise ValueError(f"Question #{i} has invalid difficulty")

                validated.append(
                    {
                        "question_id": question_id,
                        "content": content,
                        "intent": intent,
                        "rubric_ref": str(rubric_ref).strip() if rubric_ref else None,
                        "difficulty": difficulty,  # type: ignore[typeddict-item]
                    }
                )

            # Cảnh báo bài nộp không đạt KHÔNG được ghép vào content của câu hỏi: content này
            # được lưu lại và hiển thị nguyên văn ở báo cáo chấm điểm (question_content), ghép
            # vào đây thì báo cáo hiện ra một câu hỏi dính cả đoạn cảnh báo. Cảnh báo chỉ xuất
            # hiện ở lời chào đầu buổi vấn đáp (InterviewAgent._ask_question).
            next_state["questions"] = validated
            next_state["generation_status"] = "success"
            next_state["error_message"] = None
            return next_state
        except Exception as exc:
            logger.warning("validate_questions failed: %s", exc)
            next_state["questions"] = []
            next_state["error_message"] = str(exc)
            if state.get("retry_count", 0) >= 2:
                next_state["generation_status"] = "failed"
            return next_state

    def _route_after_validation(self, state: QuestionGeneratorState) -> Literal["regenerate", "finalize"]:
        if state.get("generation_status") == "success":
            return "finalize"
        if state.get("retry_count", 0) < 2:
            return "regenerate"
        return "finalize"

    def _regenerate_questions(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        next_state = dict(state)
        try:
            next_state["retry_count"] = state.get("retry_count", 0) + 1
            return next_state
        except Exception as exc:
            logger.error("regenerate_questions failed: %s", exc, exc_info=True)
            next_state["generation_status"] = "failed"
            next_state["error_message"] = str(exc)
            return next_state

    def _finalize(self, state: QuestionGeneratorState) -> QuestionGeneratorState:
        next_state = dict(state)
        if next_state.get("generation_status") != "success":
            next_state["generation_status"] = "failed"
            next_state.setdefault("error_message", "Failed to generate valid questions")
        return next_state

    def generate(
        self,
        project_description: str,
        rubric: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        validate_submission: bool = True,
    ) -> tuple[list[Question], SubmissionValidationResult]:
        init_state: QuestionGeneratorState = {
            "project_description": project_description,
            "rubric": rubric,
            "student_submission": student_submission,
            "raw_llm_output": "",
            "questions": [],
            "generation_status": "pending",
            "error_message": None,
            "retry_count": 0,
            "input_valid": False,
            "validate_submission": validate_submission,
            "submission_valid": True,
            "submission_validity_score": 0,
            "submission_status": "unknown",
            "submission_issues": [],
            "submission_summary": "",
        }

        final_state = self.graph.invoke(init_state)
        if final_state.get("generation_status") != "success":
            raise ValueError(final_state.get("error_message") or "Question generation failed")

        status: ValidationStatus = final_state.get("submission_status", "unknown")
        validation_result: SubmissionValidationResult = {
            "is_valid": status == "valid",
            "validity_score": int(final_state.get("submission_validity_score", 0)),
            "status": status,
            "verdict_label": VERDICT_LABELS[status],
            "issues": list(final_state.get("submission_issues", [])),
            "summary": str(final_state.get("submission_summary") or ""),
        }
        return final_state["questions"], validation_result


# ---------------------------------------------------------------------------
# Interview Agent
# ---------------------------------------------------------------------------

ResponseLabel = Literal[
    "answered_well",
    "partial",
    "unclear",
    "asking_clarify",
    "off_topic",
    "dont_know",
    "requesting_hint",
]


class InterviewState(TypedDict):
    questions: list[Question]
    student_name: Optional[str]
    student_submission: Optional[StudentSubmission]
    submission_status: ValidationStatus
    submission_validity_score: int
    submission_issues: list[str]
    submission_summary: str

    current_question_index: int
    interview_status: Literal["not_started", "in_progress", "completed"]

    current_question_turn_count: int
    current_question_status: Literal[
        "asking", "clarifying", "probing", "concluded_pass", "concluded_fail"
    ]

    full_chat_log: list[ChatTurn]
    current_question_chat: list[ChatTurn]

    max_turns_per_question: int
    min_turns_before_conclude: int
    hint_count: int
    max_hints_per_question: int

    student_response: str
    response_label: ResponseLabel
    last_agent_message: str
    current_question_id: str
    error_message: Optional[str]


class InterviewAgent:
    """Interactive interview graph with LangGraph interrupt/resume."""

    def __init__(self, llm_client: OpenAI, model: str) -> None:
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = _read_prompt_file(
            "interview_system_prompt.txt",
            "Bạn là gia sư vấn đáp. Đặt câu hỏi ngắn gọn, không tiết lộ đáp án."
        )
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(InterviewState)

        workflow.add_node("start_interview", self._start_interview)
        workflow.add_node("ask_question", self._ask_question)
        workflow.add_node("wait_for_student", self._wait_for_student)
        workflow.add_node("classify_response", self._classify_response)
        workflow.add_node("probe_or_clarify", self._probe_or_clarify)
        workflow.add_node("clarify_question", self._clarify_question)
        workflow.add_node("provide_hint", self._provide_hint)
        workflow.add_node("conclude_pass", self._conclude_pass)
        workflow.add_node("conclude_fail", self._conclude_fail)
        workflow.add_node("next_question_or_end", self._next_question_or_end)

        workflow.set_entry_point("start_interview")
        workflow.add_edge("start_interview", "ask_question")
        workflow.add_edge("ask_question", "wait_for_student")
        workflow.add_edge("wait_for_student", "classify_response")
        workflow.add_conditional_edges(
            "classify_response",
            self._route_after_classification,
            {
                "conclude_pass": "conclude_pass",
                "conclude_fail": "conclude_fail",
                "probe_or_clarify": "probe_or_clarify",
                "clarify_question": "clarify_question",
                "provide_hint": "provide_hint",
            },
        )
        workflow.add_edge("probe_or_clarify", "wait_for_student")
        workflow.add_edge("clarify_question", "wait_for_student")
        workflow.add_edge("provide_hint", "wait_for_student")
        workflow.add_edge("conclude_pass", "next_question_or_end")
        workflow.add_edge("conclude_fail", "next_question_or_end")
        workflow.add_conditional_edges(
            "next_question_or_end",
            self._route_after_next_question,
            {"continue": "ask_question", "end": END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def _start_interview(self, state: InterviewState) -> InterviewState:
        try:
            next_state = dict(state)
            next_state["interview_status"] = "in_progress"
            next_state["current_question_index"] = 0
            next_state["current_question_turn_count"] = 0
            next_state["current_question_status"] = "asking"
            next_state.setdefault("student_submission", None)
            next_state.setdefault("submission_status", "unknown")
            next_state.setdefault("submission_validity_score", 0)
            next_state.setdefault("submission_issues", [])
            next_state.setdefault("submission_summary", "")
            next_state["full_chat_log"] = []
            next_state["current_question_chat"] = []
            next_state.setdefault("hint_count", 0)
            next_state.setdefault("max_hints_per_question", 2)
            next_state["student_response"] = ""
            next_state["response_label"] = "unclear"
            next_state["last_agent_message"] = ""
            next_state["current_question_id"] = ""
            next_state["error_message"] = None
            return next_state
        except Exception as exc:
            logger.error("start_interview failed: %s", exc, exc_info=True)
            next_state = dict(state)
            next_state["error_message"] = str(exc)
            return next_state

    def _ask_question(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            questions = state.get("questions", [])
            q_idx = state.get("current_question_index", 0)
            if q_idx >= len(questions):
                next_state["interview_status"] = "completed"
                return next_state

            question = questions[q_idx]
            question_id = question["question_id"]
            total = len(questions)
            student_name = state.get("student_name")

            if student_name:
                prefix = f"{student_name}, "
            else:
                prefix = ""

            question_msg = f"{prefix}Câu hỏi {q_idx + 1}/{total}: {question['content']}"
            previous_status = state.get("current_question_status")
            previous_message = (state.get("last_agent_message") or "").strip()

            # Preserve conclusion/explanation before asking the next question.
            if previous_status in {"concluded_fail", "concluded_pass"} and previous_message:
                msg = f"{previous_message}\n\n---\n\n{question_msg}"
            else:
                msg = question_msg

            # Cảnh báo chỉ hiện một lần, ngay trước câu hỏi đầu tiên, để học sinh biết bài nộp
            # của mình đang bị đánh giá là sai đề TRƯỚC khi bước vào vấn đáp.
            if q_idx == 0 and state.get("submission_status") == "invalid":
                warning = _build_submission_warning_message(
                    issues=state.get("submission_issues", []),
                    validity_score=int(state.get("submission_validity_score", 0)),
                )
                msg = f"{warning}\n\n{msg}"

            next_state["current_question_status"] = "asking"
            next_state["current_question_id"] = question_id
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            return next_state
        except Exception as exc:
            logger.error("ask_question failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _wait_for_student(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            student_response = interrupt("waiting_for_student")
        except GraphInterrupt:
            # interrupt() intentionally raises GraphInterrupt to pause execution.
            raise
        except Exception as exc:
            logger.error("wait_for_student failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

        try:
            response_text = str(student_response or "").strip()
            question_id = state.get("current_question_id")

            next_state["student_response"] = response_text
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []),
                role="student",
                content=response_text,
                question_id=question_id,
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []),
                role="student",
                content=response_text,
                question_id=question_id,
            )
            return next_state
        except Exception as exc:
            logger.error("wait_for_student failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _classify_response(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            student_response = state.get("student_response", "")
            turns_text = _format_turns(state.get("current_question_chat", []))

            classify_prompt = (
                f"{self.system_prompt}\n\n"
                "Bạn chỉ được trả về một nhãn duy nhất trong danh sách:\n"
                "answered_well | partial | unclear | asking_clarify | off_topic | dont_know | requesting_hint\n\n"
                "Nếu học sinh đang xin gợi ý hoặc trợ giúp để trả lời câu hỏi, trả về requesting_hint.\n\n"
                f"CÂU HỎI: {question['content']}\n"
                f"MỤC TIÊU: {question['intent']}\n"
                f"HỘI THOẠI HIỆN TẠI:\n{turns_text}\n\n"
                f"TIN NHẮN HỌC SINH MỚI NHẤT: {student_response}\n"
                "Chỉ trả về nhãn."
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là bộ phận phân loại hội đáp."},
                    {"role": "user", "content": classify_prompt},
                ],
                temperature=0,
                max_tokens=20,
            )
            raw_label = (response.choices[0].message.content or "").strip().lower()
            label = _normalize_label(raw_label)
            next_state["response_label"] = label

            if label not in {"asking_clarify", "requesting_hint"}:
                next_state["current_question_turn_count"] = state.get("current_question_turn_count", 0) + 1
            return next_state
        except Exception as exc:
            logger.error("classify_response failed: %s", exc, exc_info=True)
            next_state["response_label"] = "unclear"
            next_state["error_message"] = str(exc)
            next_state["current_question_turn_count"] = state.get("current_question_turn_count", 0) + 1
            return next_state

    def _route_after_classification(
        self, state: InterviewState
    ) -> Literal["conclude_pass", "conclude_fail", "probe_or_clarify", "clarify_question", "provide_hint"]:
        label = state.get("response_label", "unclear")
        turn_count = state.get("current_question_turn_count", 0)
        min_turns = state.get("min_turns_before_conclude", 2)
        max_turns = state.get("max_turns_per_question", 6)

        if label == "answered_well":
            return "conclude_pass"
        if label == "requesting_hint":
            return "provide_hint"
        if label == "asking_clarify":
            return "clarify_question"
        if label == "partial" and turn_count >= min_turns:
            return "conclude_fail"
        if label == "dont_know" and turn_count >= min_turns:
            return "conclude_fail"
        if turn_count >= max_turns:
            return "conclude_fail"
        return "probe_or_clarify"

    def _probe_or_clarify(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            label = state.get("response_label", "unclear")
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            question_id = state.get("current_question_id")
            student_response = state.get("student_response", "")

            if label == "partial":
                behavior = "Bạn nói em đã đúng một phần, rồi hỏi tiếp phần còn thiếu."
            elif label == "unclear":
                behavior = "Bạn xác nhận lại ý em và yêu cầu em nói rõ hơn."
            elif label == "dont_know":
                behavior = "Bạn đưa gợi ý nhẹ bằng câu hỏi dẫn dắt, không đưa đáp án trực tiếp."
            elif label == "off_topic":
                behavior = "Bạn nhẹ nhàng kéo em quay lại trọng tâm câu hỏi hiện tại."
            else:
                behavior = "Bạn tiếp tục hỏi đào sâu thêm một ý nhỏ."

            submission_context = _build_submission_context(state.get("student_submission"))

            user_prompt = (
                f"{submission_context}\n\n"
                f"CÂU HỎI: {question['content']}\n"
                f"MỤC TIÊU: {question['intent']}\n"
                f"TIN NHẮN HỌC SINH: {student_response}\n"
                f"YÊU CẦU HÀNH VI: {behavior}\n"
                "Nếu là bài code, dẫn chiếu đến dòng/đoạn code cụ thể trong bài làm.\n"
                "ĐỪNG hỏi lý do nghiệp vụ chung chung.\n"
                "Trả về 1-2 câu ngắn bằng tiếng Việt có dấu."
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=220,
            )
            msg = (response.choices[0].message.content or "").strip()
            next_state["current_question_status"] = "probing"
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            return next_state
        except Exception as exc:
            logger.error("probe_or_clarify failed: %s", exc, exc_info=True)
            fallback = "Em thử nói rõ hơn một ý cụ thể trong câu trả lời của em được không?"
            question_id = state.get("current_question_id")
            next_state["current_question_status"] = "probing"
            next_state["last_agent_message"] = fallback
            next_state["error_message"] = str(exc)
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", fallback, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", fallback, question_id
            )
            return next_state

    def _provide_hint(self, state: InterviewState) -> InterviewState:
        """Generate a Socratic hint that guides without revealing the answer."""
        next_state = dict(state)
        try:
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            question_id = state.get("current_question_id")
            hint_count = state.get("hint_count", 0)
            max_hints = state.get("max_hints_per_question", 2)

            if hint_count >= max_hints:
                msg = "Em đã dùng hết gợi ý cho câu này rồi. Thử tự suy nghĩ thêm nhé."
                next_state["last_agent_message"] = msg
                next_state["full_chat_log"] = _append_chat_turn(
                    state.get("full_chat_log", []), "agent", msg, question_id
                )
                next_state["current_question_chat"] = _append_chat_turn(
                    state.get("current_question_chat", []), "agent", msg, question_id
                )
                return next_state

            submission_context = _build_submission_context(state.get("student_submission"))

            if hint_count == max_hints - 1:
                hint_level = (
                    "Đây là gợi ý cuối. Có thể rõ hơn một chút nhưng "
                    "TUYỆT ĐỐI không nói thẳng đáp án."
                )
            else:
                hint_level = (
                    "Dùng câu hỏi Socratic để dẫn học sinh đến câu trả lời. "
                    "KHÔNG nói thẳng đáp án."
                )

            user_prompt = (
                f"{submission_context}\n\n"
                f"CÂU HỎI ĐANG HỎI: {question['content']}\n"
                f"MỤC TIÊU: {question['intent']}\n\n"
                f"LỊCH SỬ HỘI THOẠI:\n{_format_turns(state.get('current_question_chat', []))}\n\n"
                f"YÊU CẦU HINT (lần {hint_count + 1}/{max_hints}): {hint_level}\n"
                "Nếu là bài code, dẫn chiếu đến dòng code cụ thể trong bài làm.\n"
                "Trả về 1-2 câu hint bằng tiếng Việt có dấu."
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            msg = (response.choices[0].message.content or "").strip()

            # Hint không tính vào turn_count.
            next_state["hint_count"] = hint_count + 1
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []),
                role="agent",
                content=msg,
                question_id=question_id,
            )
            return next_state
        except Exception as exc:
            logger.error("provide_hint failed: %s", exc, exc_info=True)
            fallback = "Thử xem lại phần điều kiện trong bài làm của em, có gì đặc biệt ở đó không?"
            question_id = state.get("current_question_id")
            next_state["last_agent_message"] = fallback
            next_state["error_message"] = str(exc)
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", fallback, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", fallback, question_id
            )
            return next_state

    def _clarify_question(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            question_id = state.get("current_question_id")
            student_response = state.get("student_response", "")

            submission_context = _build_submission_context(state.get("student_submission"))

            user_prompt = (
                f"{submission_context}\n\n"
                "Học sinh đang hỏi lại đề bài. Hãy diễn đạt lại câu hỏi theo cách khác,"
                " rõ ràng hơn, ngắn gọn hơn, và không tiết lộ đáp án.\n"
                f"CÂU HỎI GỐC: {question['content']}\n"
                f"MỤC TIÊU: {question['intent']}\n"
                f"PHẢN HỒI CỦA HỌC SINH: {student_response}\n"
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=220,
            )
            msg = (response.choices[0].message.content or "").strip()
            next_state["current_question_status"] = "clarifying"
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", msg, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", msg, question_id
            )
            return next_state
        except Exception as exc:
            logger.error("clarify_question failed: %s", exc, exc_info=True)
            fallback = "Thầy hỏi lại ngắn gọn: em mô tả cách em giải quyết vấn đề này trong project của em như thế nào?"
            question_id = state.get("current_question_id")
            next_state["current_question_status"] = "clarifying"
            next_state["last_agent_message"] = fallback
            next_state["error_message"] = str(exc)
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", fallback, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", fallback, question_id
            )
            return next_state

    def _conclude_pass(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            question_id = state.get("current_question_id")
            summary = self._generate_pass_summary(state)
            msg = f"✅ Tốt, em đã trả lời đúng ý chính!\n\n{summary}"

            q_idx = state.get("current_question_index", 0)
            total = len(state.get("questions", []))
            if q_idx + 1 >= total:
                next_state["interview_status"] = "completed"

            next_state["current_question_status"] = "concluded_pass"
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", msg, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", msg, question_id
            )
            return next_state
        except Exception as exc:
            logger.error("conclude_pass failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _conclude_fail(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            question_id = state.get("current_question_id")
            solved_answer = self._generate_full_explanation(state)
            msg = (
                "Không sao, câu này thầy ghi nhận đến đây.\n\n"
                f"📖 Giải đáp:\n{solved_answer}"
            )

            q_idx = state.get("current_question_index", 0)
            total = len(state.get("questions", []))
            if q_idx + 1 >= total:
                next_state["interview_status"] = "completed"

            next_state["current_question_status"] = "concluded_fail"
            next_state["last_agent_message"] = msg
            next_state["full_chat_log"] = _append_chat_turn(
                state.get("full_chat_log", []), "agent", msg, question_id
            )
            next_state["current_question_chat"] = _append_chat_turn(
                state.get("current_question_chat", []), "agent", msg, question_id
            )
            return next_state
        except Exception as exc:
            logger.error("conclude_fail failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _generate_full_explanation(self, state: InterviewState) -> str:
        try:
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            turns_text = _format_turns(state.get("current_question_chat", []))
            submission_text = _build_submission_hint_context(state.get("student_submission"))

            prompt = (
                "Bạn là giáo viên đang tổng kết và giải đáp cho học sinh sau khi vấn đáp.\n\n"
                f"CÂU HỎI: {question['content']}\n"
                f"MỤC TIÊU CÂU HỎI: {question['intent']}\n\n"
                f"BÀI LÀM HỌC SINH:\n{submission_text}\n\n"
                f"HỘI THOẠI VỪA QUA:\n{turns_text}\n\n"
                "YÊU CẦU:\n"
                "- Trả lời đầy đủ, rõ ràng cho chính câu hỏi này\n"
                "- Chỉ ra điểm học sinh thiếu/sai một cách dễ hiểu\n"
                "- Nếu có thể, liên hệ trực tiếp đến logic trong bài làm của học sinh\n"
                "- Viết 3-5 câu tiếng Việt có dấu, không quá dài dòng\n"
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là giáo viên giải đáp kiến thức cho học sinh."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=280,
            )

            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception as exc:
            logger.warning("generate_full_explanation failed: %s", exc)

        return "Đáp án đầy đủ cần bám sát mục tiêu câu hỏi và giải thích rõ logic cốt lõi; em hãy đối chiếu lại phần em đã nêu với mục tiêu để hoàn thiện kiến thức."

    def _generate_pass_summary(self, state: InterviewState) -> str:
        try:
            q_idx = state.get("current_question_index", 0)
            question = state.get("questions", [])[q_idx]
            turns_text = _format_turns(state.get("current_question_chat", []))

            prompt = (
                "Học sinh vừa trả lời đúng câu hỏi vấn đáp. Hãy viết 2-3 câu tóm tắt:\n"
                "- Điểm học sinh nêu được chính xác là gì\n"
                "- Nếu có thể, bổ sung thêm 1 chi tiết nhỏ để học sinh hiểu sâu hơn (không bắt buộc)\n"
                "Giọng văn ngắn gọn, khích lệ, bằng tiếng Việt có dấu.\n\n"
                f"CÂU HỎI: {question['content']}\n"
                f"MỤC TIÊU: {question['intent']}\n"
                f"HỘI THOẠI:\n{turns_text}"
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là giáo viên tóm tắt điểm đúng của học sinh."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=180,
            )

            result = (response.choices[0].message.content or "").strip()
            return result if result else "Em đã nắm đúng ý chính của câu hỏi này."
        except Exception as exc:
            logger.warning("_generate_pass_summary failed: %s", exc)

        return "Em đã nắm đúng ý chính của câu hỏi này."

    def _next_question_or_end(self, state: InterviewState) -> InterviewState:
        next_state = dict(state)
        try:
            q_idx = state.get("current_question_index", 0)
            total = len(state.get("questions", []))

            if q_idx + 1 >= total:
                next_state["interview_status"] = "completed"
                return next_state

            next_state["current_question_index"] = q_idx + 1
            next_state["current_question_turn_count"] = 0
            next_state["current_question_chat"] = []
            next_state["current_question_status"] = "asking"
            next_state["current_question_id"] = ""
            next_state["hint_count"] = 0
            return next_state
        except Exception as exc:
            logger.error("next_question_or_end failed: %s", exc, exc_info=True)
            next_state["error_message"] = str(exc)
            return next_state

    def _route_after_next_question(self, state: InterviewState) -> Literal["continue", "end"]:
        if state.get("interview_status") == "completed":
            return "end"
        return "continue"

    def start_session(
        self,
        session_id: str,
        questions: list[Question],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        submission_status: ValidationStatus = "unknown",
        submission_validity_score: int = 0,
        submission_issues: Optional[list[str]] = None,
        submission_summary: str = "",
        max_turns_per_question: int = 6,
        min_turns_before_conclude: int = 2,
        max_hints_per_question: int = 2,
    ) -> InterviewState:
        initial_state: InterviewState = {
            "questions": questions,
            "student_name": student_name,
            "student_submission": student_submission,
            "submission_status": submission_status,
            "submission_validity_score": submission_validity_score,
            "submission_issues": submission_issues or [],
            "submission_summary": submission_summary,
            "current_question_index": 0,
            "interview_status": "not_started",
            "current_question_turn_count": 0,
            "current_question_status": "asking",
            "full_chat_log": [],
            "current_question_chat": [],
            "max_turns_per_question": max_turns_per_question,
            "min_turns_before_conclude": min_turns_before_conclude,
            "hint_count": 0,
            "max_hints_per_question": max_hints_per_question,
            "student_response": "",
            "response_label": "unclear",
            "last_agent_message": "",
            "current_question_id": "",
            "error_message": None,
        }

        config = {"configurable": {"thread_id": session_id}}
        self.graph.invoke(initial_state, config=config)
        return self.get_session_state(session_id)

    def resume_session(self, session_id: str, student_message: str) -> InterviewState:
        config = {"configurable": {"thread_id": session_id}}
        self.graph.invoke(Command(resume=student_message), config=config)
        return self.get_session_state(session_id)

    def get_session_state(self, session_id: str) -> InterviewState:
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        values = snapshot.values if snapshot else None
        if not values:
            raise ValueError(f"Interview session not found: {session_id}")
        return dict(values)

    def delete_session(self, session_id: str) -> None:
        """Xoá checkpoint của một thread. Không có thì thôi, không ném lỗi."""
        try:
            self.checkpointer.delete_thread(session_id)
        except AttributeError:
            # MemorySaver bản cũ chưa có delete_thread → dọn tay bằng cấu trúc nội bộ.
            for attr in ("storage", "writes", "blobs"):
                store = getattr(self.checkpointer, attr, None)
                if isinstance(store, dict):
                    store.pop(session_id, None)
        except Exception as exc:
            logger.warning("delete_session failed for %s: %s", session_id, exc)


# ---------------------------------------------------------------------------
# Grading Agent
# ---------------------------------------------------------------------------

class GradingState(TypedDict):
    questions: list[Question]
    full_chat_log: list[ChatTurn]
    rubric: Optional[str]
    student_name: Optional[str]
    student_submission: Optional[StudentSubmission]
    submission_status: ValidationStatus
    submission_validity_score: int
    submission_issues: list[str]
    submission_summary: str

    per_question_logs: dict[str, list[ChatTurn]]
    grading_status: Literal["pending", "in_progress", "completed", "failed"]

    scores: list[QuestionScore]
    final_report: Optional[GradingReport]
    error_message: Optional[str]


class GradingAgent:
    """Grade each question based on segmented chat logs."""

    SCORE_LABELS = {
        0: "không hiểu",
        1: "hiểu một phần",
        2: "hiểu hoàn toàn",
    }

    def __init__(self, llm_client: OpenAI, model: str) -> None:
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = _read_prompt_file(
            "grading_system_prompt.txt",
            "Ban la giao vien cham diem van dap."
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GradingState)
        workflow.add_node("segment_log_by_question", self._segment_log_by_question)
        workflow.add_node("grade_each_question", self._grade_each_question)
        workflow.add_node("validate_scores", self._validate_scores)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("finalize_report", self._finalize_report)

        workflow.set_entry_point("segment_log_by_question")
        workflow.add_edge("segment_log_by_question", "grade_each_question")
        workflow.add_edge("grade_each_question", "validate_scores")
        workflow.add_edge("validate_scores", "generate_summary")
        workflow.add_edge("generate_summary", "finalize_report")
        workflow.add_edge("finalize_report", END)
        return workflow.compile()

    def _segment_log_by_question(self, state: GradingState) -> GradingState:
        next_state = dict(state)
        try:
            segmented: dict[str, list[ChatTurn]] = {}
            for turn in state.get("full_chat_log", []):
                question_id = turn.get("question_id")
                if not question_id:
                    continue
                segmented.setdefault(question_id, []).append(turn)

            next_state["per_question_logs"] = segmented
            next_state["grading_status"] = "in_progress"
            next_state["error_message"] = None
            return next_state
        except Exception as exc:
            logger.error("segment_log_by_question failed: %s", exc, exc_info=True)
            next_state["grading_status"] = "failed"
            next_state["error_message"] = str(exc)
            return next_state

    def _grade_each_question(self, state: GradingState) -> GradingState:
        next_state = dict(state)
        scores: list[QuestionScore] = []
        try:
            questions = state.get("questions", [])
            segmented_logs = state.get("per_question_logs", {})

            # Chỉ ép 0 khi đã kết luận CHẮC CHẮN là bài nộp sai đề. Trạng thái "unknown"
            # (không có bài nộp / không lấy được đề bài / lỗi hệ thống) không được ép 0 —
            # học sinh không chịu trách nhiệm cho việc hệ thống không đối chiếu được.
            force_zero = state.get("submission_status") == "invalid"

            for question in questions:
                qid = question["question_id"]
                logs = segmented_logs.get(qid, [])
                formatted_chat_log = _format_turns(logs)
                rubric_line = f"RUBRIC: {question.get('rubric_ref')}" if question.get("rubric_ref") else ""

                submission_context = ""
                if state.get("student_submission"):
                    sub = state["student_submission"]
                    code_or_text = sub.get("code") or sub.get("text") or ""
                    lang = sub.get("language") or ""
                    if code_or_text:
                        submission_context = f"\nBÀI LÀM HỌC SINH ({lang}):\n```{lang}\n{code_or_text}\n```\n"

                submission_warning = ""
                if force_zero:
                    issues_str = "; ".join(state.get("submission_issues", [])) or "không bám đề bài"
                    submission_warning = (
                        f"\nLƯU Ý: Bài nộp đã bị đánh giá KHÔNG ĐẠT yêu cầu đề bài "
                        f"(điểm phù hợp {state.get('submission_validity_score', 0)}/100). "
                        f"Vấn đề: {issues_str}. "
                        "Hãy nhận xét dựa trên đề bài, và nêu rõ trong what_student_missed "
                        "rằng bài nộp không đáp ứng yêu cầu đề bài.\n"
                    )

                prompt = (
                    "Bạn là giáo viên chấm điểm vấn đáp. Hãy đánh giá câu trả lời của học sinh.\n\n"
                    f"CÂU HỎI: {question['content']}\n"
                    f"MỤC TIÊU CÂU HỎI: {question['intent']}\n"
                    f"{rubric_line}\n"
                    f"{submission_context}"
                    f"{submission_warning}\n"
                    f"LỊCH SỬ HỘI THOẠI:\n{formatted_chat_log}\n\n"
                    "THANG ĐIỂM:\n"
                    "- 0 (không hiểu): Không trả lời được, sai hoàn toàn, hoặc chỉ đoán mò\n"
                    "- 1 (hiểu một phần): Nắm được ý chính nhưng thiếu chi tiết hoặc có sai sót\n"
                    "- 2 (hiểu hoàn toàn): Trả lời đúng và đủ, thể hiện hiểu bản chất\n\n"
                    "Trả về JSON với các field:\n"
                    "- score: 0, 1, hoặc 2\n"
                    "- score_label: 'không hiểu' | 'hiểu một phần' | 'hiểu hoàn toàn'\n"
                    "- what_student_got_right: học sinh trả lời đúng gì\n"
                    "- what_student_missed: còn thiếu gì để đạt điểm cao hơn\n"
                    "- key_evidence: trích dẫn 1-2 câu từ log thể hiện rõ nhất mức độ hiểu\n"
                    "- rubric_criteria_met: tiêu chí rubric đã đạt (null nếu không có rubric)\n"
                )

                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                )
                payload = _extract_json_object((response.choices[0].message.content or "").strip())

                score_val = int(payload.get("score", 0))
                if score_val not in (0, 1, 2):
                    score_val = min(max(score_val, 0), 2)

                what_missed = str(payload.get("what_student_missed") or "").strip()

                if force_zero:
                    score_val = 0
                    # Nói thẳng lý do bị 0, nếu không học sinh chỉ thấy điểm 0 mà không hiểu vì sao.
                    forced_note = "Bài nộp không đáp ứng yêu cầu đề bài nên câu này được tính 0 điểm."
                    what_missed = f"{forced_note} {what_missed}".strip()

                score_label = payload.get("score_label") or self.SCORE_LABELS[score_val]
                normalized_label = _normalize_score_label(str(score_label), score_val)

                scores.append(
                    {
                        "question_id": qid,
                        "question_content": question["content"],
                        "score": score_val,  # type: ignore[typeddict-item]
                        "score_label": normalized_label,  # type: ignore[typeddict-item]
                        "what_student_got_right": str(payload.get("what_student_got_right") or "").strip(),
                        "what_student_missed": what_missed,
                        "key_evidence": str(payload.get("key_evidence") or "").strip(),
                        "rubric_criteria_met": (
                            str(payload.get("rubric_criteria_met")).strip()
                            if payload.get("rubric_criteria_met")
                            else None
                        ),
                    }
                )

            next_state["scores"] = scores
            return next_state
        except Exception as exc:
            logger.error("grade_each_question failed: %s", exc, exc_info=True)
            next_state["scores"] = scores
            next_state["grading_status"] = "failed"
            next_state["error_message"] = str(exc)
            return next_state

    def _validate_scores(self, state: GradingState) -> GradingState:
        next_state = dict(state)
        try:
            questions = state.get("questions", [])
            scores = state.get("scores", [])

            score_map = {item["question_id"]: item for item in scores}
            completed_scores: list[QuestionScore] = []

            for question in questions:
                qid = question["question_id"]
                if qid in score_map:
                    completed_scores.append(score_map[qid])
                else:
                    completed_scores.append(
                        {
                            "question_id": qid,
                            "question_content": question["content"],
                            "score": 0,
                            "score_label": "không hiểu",
                            "what_student_got_right": "",
                            "what_student_missed": "Khong du du lieu hoi dap cho cau nay.",
                            "key_evidence": "",
                            "rubric_criteria_met": None,
                        }
                    )

            next_state["scores"] = completed_scores
            return next_state
        except Exception as exc:
            logger.error("validate_scores failed: %s", exc, exc_info=True)
            next_state["grading_status"] = "failed"
            next_state["error_message"] = str(exc)
            return next_state

    def _generate_summary(self, state: GradingState) -> GradingState:
        next_state = dict(state)
        scores = state.get("scores", [])
        total_score = sum(item["score"] for item in scores)
        # Điểm tối đa thực tế = số câu × 2 (thang 0/1/2), KHÔNG hardcode 10.
        max_score = (len(scores) * PER_QUESTION_MAX) or 1
        percentage = round((total_score / max_score) * 100, 2)
        score_out_of_10 = round((total_score / max_score) * 10, 2)  # quy về thang 10

        status: ValidationStatus = state.get("submission_status", "unknown")
        verdict = VERDICT_LABELS[status]
        validity_score = int(state.get("submission_validity_score", 0))
        issues = list(state.get("submission_issues", []))

        def build_report(summary: str) -> GradingReport:
            return {
                "student_name": state.get("student_name"),
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "score_out_of_10": score_out_of_10,
                "summary": summary,
                "per_question": scores,
                "generated_at": _utc_now_iso(),
                "submission_status": status,
                "submission_verdict": verdict,
                "submission_valid": status == "valid",
                "submission_validity_score": validity_score,
                "submission_issues": issues,
                "submission_summary": str(state.get("submission_summary") or ""),
                "forced_zero": status == "invalid",
            }

        try:
            if status == "invalid":
                verdict_line = (
                    f"KẾT LUẬN VỀ BÀI NỘP: {verdict} (điểm phù hợp với đề bài "
                    f"{validity_score}/100). Vấn đề: {'; '.join(issues) or 'không bám đề bài'}. "
                    "Toàn bộ câu hỏi bị tính 0 điểm vì lý do này — hãy nêu rõ điều đó trong "
                    "nhận xét và hướng dẫn học sinh nộp lại đúng đề."
                )
            elif status == "valid":
                verdict_line = (
                    f"KẾT LUẬN VỀ BÀI NỘP: {verdict} "
                    f"(điểm phù hợp với đề bài {validity_score}/100)."
                )
            else:
                verdict_line = (
                    f"KẾT LUẬN VỀ BÀI NỘP: {verdict}. Không đối chiếu được bài nộp với đề bài, "
                    "chỉ nhận xét dựa trên phần vấn đáp."
                )

            summary_prompt = (
                "Tóm tắt ngắn gọn kết quả vấn đáp của học sinh (2-4 câu, tiếng Việt có dấu).\n"
                f"{verdict_line}\n"
                f"Tổng điểm: {total_score}/{max_score} ({percentage}%, ~{score_out_of_10}/10).\n"
                f"Chi tiết: {json.dumps(scores, ensure_ascii=False)}"
            )

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là giáo viên tổng kết kết quả học tập."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.2,
                max_tokens=320,
            )

            summary = (response.choices[0].message.content or "").strip()
            next_state["final_report"] = build_report(summary or "Đã hoàn tất chấm điểm vấn đáp.")
            return next_state
        except Exception as exc:
            logger.error("generate_summary failed: %s", exc, exc_info=True)
            next_state["final_report"] = build_report("Đã hoàn tất chấm điểm vấn đáp.")
            next_state["error_message"] = str(exc)
            return next_state

    def _finalize_report(self, state: GradingState) -> GradingState:
        next_state = dict(state)
        if next_state.get("final_report") is None:
            next_state["grading_status"] = "failed"
            next_state.setdefault("error_message", "Missing final_report")
            return next_state
        next_state["grading_status"] = "completed"
        return next_state

    def grade(
        self,
        questions: list[Question],
        full_chat_log: list[ChatTurn],
        rubric: Optional[str],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        submission_status: ValidationStatus = "unknown",
        submission_validity_score: int = 0,
        submission_issues: Optional[list[str]] = None,
        submission_summary: str = "",
    ) -> GradingReport:
        initial_state: GradingState = {
            "questions": questions,
            "full_chat_log": full_chat_log,
            "rubric": rubric,
            "student_name": student_name,
            "student_submission": student_submission,
            "submission_status": submission_status,
            "submission_validity_score": submission_validity_score,
            "submission_issues": submission_issues or [],
            "submission_summary": submission_summary,
            "per_question_logs": {},
            "grading_status": "pending",
            "scores": [],
            "final_report": None,
            "error_message": None,
        }

        final_state = self.graph.invoke(initial_state)
        if final_state.get("grading_status") != "completed" or not final_state.get("final_report"):
            raise ValueError(final_state.get("error_message") or "Grading failed")
        return final_state["final_report"]


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class ProjectQAOrchestrator:
    """Coordinate QuestionGenerator -> Interview -> Grading pipeline."""

    def __init__(self, llm_provider: str = "open_ai") -> None:
        self.llm_client, _model = build_llm_client(llm_provider)

        default_model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        base_model = os.getenv("QA_LLM_MODEL", default_model)
        interview_model = os.getenv("QA_INTERVIEW_LLM_MODEL", "gpt-4o")

        self.question_generator = QuestionGeneratorAgent(self.llm_client, model=base_model)
        self.interview_agent = InterviewAgent(self.llm_client, model=interview_model)
        self.grading_agent = GradingAgent(self.llm_client, model=base_model)

    def generate_questions(
        self,
        project_description: str,
        rubric: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        validate_submission: bool = True,
    ) -> tuple[list[Question], SubmissionValidationResult]:
        return self.question_generator.generate(
            project_description=project_description,
            rubric=rubric,
            student_submission=student_submission,
            validate_submission=validate_submission,
        )

    def run_interview(
        self,
        questions: list[Question],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        submission_status: ValidationStatus = "unknown",
        submission_validity_score: int = 0,
        submission_issues: Optional[list[str]] = None,
        submission_summary: str = "",
    ) -> list[ChatTurn]:
        """Start interview and return current log snapshot.

        Note: Interview uses interrupt/resume. This method starts a transient session and
        returns the initial log containing the first question.
        """
        transient_session_id = f"transient-{uuid4()}"
        state = self.interview_agent.start_session(
            session_id=transient_session_id,
            questions=questions,
            student_name=student_name,
            student_submission=student_submission,
            submission_status=submission_status,
            submission_validity_score=submission_validity_score,
            submission_issues=submission_issues,
            submission_summary=submission_summary,
        )
        return state.get("full_chat_log", [])

    def start_interview_session(
        self,
        session_id: str,
        questions: list[Question],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        submission_status: ValidationStatus = "unknown",
        submission_validity_score: int = 0,
        submission_issues: Optional[list[str]] = None,
        submission_summary: str = "",
        max_turns_per_question: int = 6,
        min_turns_before_conclude: int = 2,
        max_hints_per_question: int = 2,
    ) -> InterviewState:
        return self.interview_agent.start_session(
            session_id=session_id,
            questions=questions,
            student_name=student_name,
            student_submission=student_submission,
            submission_status=submission_status,
            submission_validity_score=submission_validity_score,
            submission_issues=submission_issues,
            submission_summary=submission_summary,
            max_turns_per_question=max_turns_per_question,
            min_turns_before_conclude=min_turns_before_conclude,
            max_hints_per_question=max_hints_per_question,
        )

    def resume_interview_session(self, session_id: str, student_message: str) -> InterviewState:
        return self.interview_agent.resume_session(session_id=session_id, student_message=student_message)

    def get_interview_session_state(self, session_id: str) -> InterviewState:
        return self.interview_agent.get_session_state(session_id=session_id)

    def drop_interview_session(self, session_id: str) -> None:
        """Vứt checkpoint của một phiên vấn đáp (dùng khi học sinh làm lại từ đầu).

        MemorySaver giữ toàn bộ state trong RAM và không tự dọn, nên mỗi lần retake mà
        không xoá là một phiên chết nằm lại trong tiến trình.
        """
        self.interview_agent.delete_session(session_id)

    def grade(
        self,
        questions: list[Question],
        chat_log: list[ChatTurn],
        rubric: Optional[str],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        submission_status: ValidationStatus = "unknown",
        submission_validity_score: int = 0,
        submission_issues: Optional[list[str]] = None,
        submission_summary: str = "",
    ) -> GradingReport:
        return self.grading_agent.grade(
            questions=questions,
            full_chat_log=chat_log,
            rubric=rubric,
            student_name=student_name,
            student_submission=student_submission,
            submission_status=submission_status,
            submission_validity_score=submission_validity_score,
            submission_issues=submission_issues,
            submission_summary=submission_summary,
        )

    def run_full_pipeline(
        self,
        project_description: str,
        rubric: Optional[str],
        student_name: Optional[str],
        student_submission: Optional[StudentSubmission] = None,
        student_responses: Optional[list[str]] = None,
    ) -> GradingReport:
        """Run full pipeline in a non-interactive convenience mode.

        This utility is intended for test automation; production interview flow should
        use start_interview_session/resume_interview_session.
        """
        questions, validation = self.generate_questions(
            project_description=project_description,
            rubric=rubric,
            student_submission=student_submission,
        )
        session_id = f"pipeline-{uuid4()}"
        state = self.start_interview_session(
            session_id=session_id,
            questions=questions,
            student_name=student_name,
            student_submission=student_submission,
            submission_status=validation["status"],
            submission_validity_score=validation["validity_score"],
            submission_issues=validation["issues"],
            submission_summary=validation["summary"],
        )

        for answer in student_responses or []:
            state = self.resume_interview_session(session_id=session_id, student_message=answer)
            if state.get("interview_status") == "completed":
                break

        if state.get("interview_status") != "completed":
            raise ValueError("Interview is not completed. Provide enough student_responses to finish.")

        chat_log = state.get("full_chat_log", [])
        return self.grade(
            questions=questions,
            chat_log=chat_log,
            rubric=rubric,
            student_name=student_name,
            student_submission=student_submission,
            submission_status=validation["status"],
            submission_validity_score=validation["validity_score"],
            submission_issues=validation["issues"],
            submission_summary=validation["summary"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_conclude(state: InterviewState) -> str:
    if state["current_question_status"] == "concluded_pass":
        return "pass"
    if state["current_question_turn_count"] >= state["max_turns_per_question"]:
        return "fail"
    return "continue"


def _read_prompt_file(file_name: str, fallback: str) -> str:
    prompt_dir = Path(__file__).resolve().parent / "prompt" / "project_qa_agent"
    prompt_path = prompt_dir / file_name
    try:
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return fallback


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _append_chat_turn(
    turns: list[ChatTurn],
    role: Literal["agent", "student"],
    content: str,
    question_id: Optional[str],
) -> list[ChatTurn]:
    next_turns = list(turns)
    next_turns.append(
        {
            "role": role,
            "content": content,
            "turn_index": len(next_turns),
            "question_id": question_id or "",
            "timestamp": _utc_now_iso(),
        }
    )
    return next_turns


def _normalize_label(raw_label: str) -> ResponseLabel:
    text = raw_label.lower().strip()
    valid = {
        "answered_well",
        "partial",
        "unclear",
        "asking_clarify",
        "off_topic",
        "dont_know",
        "requesting_hint",
    }
    for label in valid:
        if label in text:
            return label  # type: ignore[return-value]

    if _is_hint_request(text):
        return "requesting_hint"

    if "clarify" in text or "hoi" in text:
        return "asking_clarify"
    if "well" in text or "tot" in text:
        return "answered_well"
    if "partial" in text or "mot phan" in text:
        return "partial"
    if "off" in text or "lech" in text:
        return "off_topic"
    if "khong biet" in text or "dont" in text:
        return "dont_know"
    return "unclear"


def _is_hint_request(text: str) -> bool:
    hint_phrases = [
        "requesting_hint",
        "xin goi y",
        "goi y",
        "hint",
        "khong biet bat dau",
        "toi khong biet bat dau",
        "khong biet bat dau tu dau",
        "giup toi",
        "giup em",
        "thay goi y",
    ]
    lowered = text.lower()
    return any(phrase in lowered for phrase in hint_phrases)


def _get_submission_content(submission: Optional[StudentSubmission]) -> str:
    if not submission:
        return ""
    return (submission.get("code") or submission.get("text") or "").strip()


def _build_submission_context_for_prompt(submission: Optional[StudentSubmission]) -> str:
    if not submission:
        return "Khong co bai lam hoc sinh."

    code_or_text = _get_submission_content(submission)
    language = submission.get("language") or "khong ro"
    notes = submission.get("notes") or "khong co"
    if not code_or_text:
        return f"Language: {language}\nNotes: {notes}\nNo code/text provided."

    return f"Language: {language}\nNotes: {notes}\nContent:\n{code_or_text}"


def _apply_validation(
    state: dict[str, Any],
    status: ValidationStatus,
    score: int,
    issues: list[str],
    summary: str,
) -> dict[str, Any]:
    """Ghi kết quả xác thực vào state. Gom một chỗ để 4 field không bao giờ lệch nhau."""
    state["submission_status"] = status
    state["submission_valid"] = status == "valid"
    state["submission_validity_score"] = score
    state["submission_issues"] = issues
    state["submission_summary"] = summary
    return state


def _build_submission_warning_message(issues: list[str], validity_score: int) -> str:
    issues_summary = "; ".join(issues[:3]) if issues else "không thấy phần nào giải quyết yêu cầu đề bài"
    return (
        "⚠️ **Bài nộp của em được đánh giá là KHÔNG ĐẠT yêu cầu đề bài** "
        f"(điểm phù hợp với đề bài: **{validity_score}/{100}**, cần tối thiểu "
        f"{SUBMISSION_VALIDITY_THRESHOLD}).\n\n"
        f"Vấn đề ghi nhận: {issues_summary}.\n\n"
        "Thầy vẫn tiến hành vấn đáp để em có cơ hội giải thích, nhưng nếu kết luận này "
        "không thay đổi thì bài sẽ bị tính 0 điểm. Nếu em nộp nhầm repository, "
        "hãy nộp lại đúng bài rồi vấn đáp lại."
    )


def _build_submission_context(submission: Optional[StudentSubmission]) -> str:
    """Build a formatted context block from student submission for prompt injection."""
    if not submission:
        return ""

    code = submission.get("code") or ""
    text = submission.get("text") or ""
    lang = submission.get("language") or "code"
    notes = submission.get("notes") or ""

    parts: list[str] = []
    if code:
        parts.append(f"BÀI LÀM CỦA HỌC SINH ({lang}):\n```{lang}\n{code}\n```")
    elif text:
        parts.append(f"BÀI LÀM CỦA HỌC SINH:\n{text}")
    if notes:
        parts.append(f"GHI CHÚ CỦA HỌC SINH: {notes}")

    if not parts:
        return ""

    return (
        "\n".join(parts) + "\n\n"
        "Khi probe hoặc hint, hãy dẫn chiếu đến phần cụ thể trong bài làm trên.\n"
        "KHÔNG hỏi lý do nghiệp vụ chung chung nếu có thể hỏi về logic trong code."
    )


def _build_submission_probe_context(submission: Optional[StudentSubmission]) -> str:
    return _build_submission_context(submission)


def _build_submission_hint_context(submission: Optional[StudentSubmission]) -> str:
    context = _build_submission_context(submission)
    if not context:
        return "(Không có bài làm đã nộp để tham chiếu)"
    return context


def _normalize_score_label(label: str, score: int) -> Literal["không hiểu", "hiểu một phần", "hiểu hoàn toàn"]:
    text = label.lower().strip()
    if "hoan" in text or "toan" in text:
        return "hiểu hoàn toàn"
    if "mot phan" in text or "partial" in text:
        return "hiểu một phần"
    if "khong" in text or "0" == text:
        return "không hiểu"

    if score == 2:
        return "hiểu hoàn toàn"
    if score == 1:
        return "hiểu một phần"
    return "không hiểu"


def _extract_json_array(raw_text: str) -> list[Any]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("Cannot parse JSON array from LLM output")


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Cannot parse JSON object from LLM output")


def _format_turns(turns: list[ChatTurn]) -> str:
    if not turns:
        return "(khong co du lieu)"
    lines = []
    for turn in turns:
        lines.append(f"- {turn['role']}: {turn['content']}")
    return "\n".join(lines)
