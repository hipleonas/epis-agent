"""Question Generator - Pre-generates the oral exam question bank from project analysis.

Design:
  One service call per session (at submission time, not at exam time).
  Two LLM requests total:
    1. project_summary  (~50 tokens out)
    2. question_bank    (all N questions in one shot, ~600 tokens out)
  The resulting question_bank is stored by the backend (PostgreSQL JSONB).
  The Q&A agent never sees the full repository JSON again.
"""
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from src.services.llm_client_factory import build_llm_client

logger = logging.getLogger(__name__)

# Số câu hỏi vấn đáp: cố định đúng 3 câu (thống nhất với QUESTION_COUNT của luồng showcase và với
# max_questions phía gsai-server). Trước đây sinh 5–7 câu nên phiên vấn đáp dài hơn thiết kế.
MAX_QUESTIONS = 3
MIN_QUESTIONS = 3

# Points available per question
MAX_SCORE_PER_QUESTION = 2


class QuestionGeneratorService:
    """Generates a pre-baked oral-exam question bank from analysis output."""

    def __init__(self, llm_provider: str = "open_ai") -> None:
        self.llm_client, self.model = build_llm_client(llm_provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, analysis: Dict[str, Any], assignment: str = "") -> Dict[str, Any]:
        """Generate question_bank and project_summary from AnalyzeAgent output.

        Args:
            analysis:   Dict returned by AnalyzeAgent.analyze().
            assignment: Raw assignment requirements text (used as fallback when
                        no functions were extracted from the codebase).

        Returns:
            {
                "question_bank":   [ QuestionBankItem-compatible dicts ],
                "project_summary": str,
            }
        """
        logger.info("📝 QuestionGeneratorService: generating question bank…")

        project_summary = self._generate_summary(analysis, assignment)
        candidates = self._select_candidates(analysis)

        if candidates:
            question_bank = self._generate_questions(candidates, project_summary)
        else:
            # No code structure found — generate questions from assignment text alone
            logger.warning(
                "No candidates from code analysis; falling back to assignment-based questions."
            )
            question_bank = self._generate_questions_from_assignment(
                assignment or "", project_summary
            )

        logger.info(
            "✅ QuestionGeneratorService: %d questions generated", len(question_bank)
        )
        return {
            "question_bank": question_bank,
            "project_summary": project_summary,
        }

    # ------------------------------------------------------------------
    # Step 1 – Project Summary
    # ------------------------------------------------------------------

    def _generate_summary(self, analysis: Dict[str, Any], assignment: str = "") -> str:
        """Produce a short (~150-word) plain-text summary of the project."""
        reqs = analysis.get("requirements", [])
        unmatched = analysis.get("unmatched_functions", [])

        if reqs:
            req_lines = "\n".join(
                f"- {r.get('requirement_text', '')} "
                f"[{', '.join(f['name'] for f in r.get('functions', [])[:3])}]"
                for r in reqs
            )
            unmatched_names = ", ".join(f.get("name", "") for f in unmatched[:5])
            content_block = (
                f"=== REQUIREMENTS MAPPED TO FUNCTIONS ===\n{req_lines}\n\n"
                f"=== HELPER / UNMATCHED FUNCTIONS ===\n{unmatched_names or 'none'}"
            )
        else:
            # Fallback: summarise from the raw assignment
            content_block = f"=== ĐỀ BÀI ===\n{assignment[:1500]}" if assignment else "(No code or assignment available)"

        prompt = (
            "You are summarising a student's programming assignment.\n\n"
            f"{content_block}\n\n"
            "Write a concise, factual summary in Vietnamese (max 150 words) describing:\n"
            "1. What the project does overall\n"
            "2. The main functions/classes implemented (if any)\n"
            "3. Any obvious gaps or issues\n\n"
            "Return ONLY the summary text, no headings."
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s — using fallback.", exc)
            return (
                f"Project có {len(analysis.get('requirements', []))} requirements được implement. "
                + (
                    "Functions chính: "
                    + ", ".join(
                        f['name']
                        for r in analysis.get('requirements', [])[:3]
                        for f in r.get('functions', [])[:2]
                    )
                    if analysis.get('requirements')
                    else (assignment[:200] if assignment else "Không có thông tin")
                )
            )

    # ------------------------------------------------------------------
    # Step 2 – Candidate Selection
    # ------------------------------------------------------------------

    def _select_candidates(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select the most exam-worthy topics from the analysis.

        Priority order:
          1. inappropriate_code on high-importance requirements  → reveals bugs
          2. implementation logic on high-importance requirements → algorithm understanding
          3. design_decision where ≥2 functions present          → architecture thinking
          4. update_suggestion                                    → improvement awareness
          5. implementation on medium-importance requirements
        """
        reqs: List[Dict] = analysis.get("requirements", [])

        candidates: List[Dict[str, Any]] = []

        def add(c: Dict) -> None:
            if len(candidates) < MAX_QUESTIONS:
                candidates.append(c)

        # --- Priority 1: bugs in high-importance requirements
        for idx, req in enumerate(reqs):
            if req.get("importance") == "high" and req.get("inappropriate_code", "").strip():
                add({
                    "type": "inappropriate_code",
                    "requirement": req.get("requirement_text", ""),
                    "functions": req.get("functions", [])[:2],
                    "detail": req["inappropriate_code"],
                    "idea": req.get("idea", ""),
                })

        # --- Priority 2: implementation logic on high-importance requirements
        for req in reqs:
            if req.get("importance") == "high":
                for func in req.get("functions", [])[:1]:
                    add({
                        "type": "implementation",
                        "requirement": req.get("requirement_text", ""),
                        "functions": [func],
                        "detail": req.get("idea", ""),
                        "idea": req.get("idea", ""),
                    })

        # --- Priority 3: design decisions (≥2 functions per requirement)
        for req in reqs:
            if len(req.get("functions", [])) >= 2:
                add({
                    "type": "design_decision",
                    "requirement": req.get("requirement_text", ""),
                    "functions": req.get("functions", [])[:3],
                    "detail": req.get("idea", ""),
                    "idea": req.get("idea", ""),
                })

        # --- Priority 4: update suggestions
        for req in reqs:
            if req.get("update_suggestion", "").strip():
                add({
                    "type": "update_suggestion",
                    "requirement": req.get("requirement_text", ""),
                    "functions": req.get("functions", [])[:2],
                    "detail": req["update_suggestion"],
                    "idea": req.get("idea", ""),
                })

        # --- Priority 5: medium-importance implementation fallback
        for req in reqs:
            if req.get("importance") != "high":
                for func in req.get("functions", [])[:1]:
                    add({
                        "type": "implementation",
                        "requirement": req.get("requirement_text", ""),
                        "functions": [func],
                        "detail": req.get("idea", ""),
                        "idea": req.get("idea", ""),
                    })

        # Ensure minimum
        if len(candidates) < MIN_QUESTIONS:
            for req in reqs:
                for func in req.get("functions", []):
                    if len(candidates) >= MIN_QUESTIONS:
                        break
                    add({
                        "type": "implementation",
                        "requirement": req.get("requirement_text", ""),
                        "functions": [func],
                        "detail": req.get("idea", ""),
                        "idea": req.get("idea", ""),
                    })

        return candidates[:MAX_QUESTIONS]

    # ------------------------------------------------------------------
    # Step 3 – Batch Question Generation (single LLM call)
    # ------------------------------------------------------------------

    def _generate_questions(
        self, candidates: List[Dict[str, Any]], project_summary: str
    ) -> List[Dict[str, Any]]:
        """Generate all questions + rubrics in one LLM call.

        Returns a list of QuestionBankItem-compatible dicts.
        """
        if not candidates:
            return []

        # Describe each candidate concisely for the prompt
        candidate_descriptions = []
        for i, c in enumerate(candidates, 1):
            func_names = ", ".join(
                f"`{f.get('name', '?')}` ({f.get('file', '?')})"
                for f in c.get("functions", [])[:2]
            )
            candidate_descriptions.append(
                f"Q{i}:\n"
                f"  type: {c['type']}\n"
                f"  requirement: {c['requirement']}\n"
                f"  functions: {func_names or 'N/A'}\n"
                f"  detail: {c['detail'][:200]}\n"
            )

        candidates_block = "\n".join(candidate_descriptions)

        prompt = f"""Bạn là giảng viên tạo bộ câu hỏi vấn đáp oral exam cho sinh viên.

=== TÓM TẮT PROJECT ===
{project_summary}

=== DANH SÁCH CHỦ ĐỀ CẦN HỎI ===
{candidates_block}

=== YÊU CẦU ===
Với mỗi chủ đề trên, tạo MỘT câu hỏi vấn đáp theo quy tắc:
- Câu hỏi bằng tiếng Việt, tự nhiên như giảng viên thực sự hỏi
- KHÔNG tiết lộ đáp án hoặc lỗi trong câu hỏi
- Với type "inappropriate_code": hỏi gián tiếp (ví dụ: "Em có thể giải thích tại sao em làm X không?")
- Với type "implementation": hỏi về logic, thuật toán, edge cases
- Với type "design_decision": hỏi về lý do kiến trúc
- Với type "update_suggestion": hỏi về cải tiến có thể có
- TẤT CẢ các trường (question, expected_concepts, scoring_rubric) đều phải bằng TIẾNG VIỆT

Trả về JSON array:
[
  {{
    "id": "q1",
    "question": "...",
    "target_function": "tên function (null nếu không có)",
    "target_file": "path/to/file.py (null nếu không có)",
    "topic_type": "inappropriate_code|implementation|design_decision|update_suggestion",
    "expected_concepts": ["khái niệm 1 bằng tiếng Việt", "khái niệm 2", "khái niệm 3"],
    "scoring_rubric": {{
      "score_2": "Tiêu chí 2 điểm (đầy đủ, chính xác)",
      "score_1": "Tiêu chí 1 điểm (một phần đúng)",
      "score_0": "Tiêu chí 0 điểm (sai hoặc không biết)"
    }},
    "max_score": 2
  }},
  ...
]

Chỉ trả về JSON array, không thêm text hay markdown."""

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON array without markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

            questions: List[Dict] = json.loads(raw)

            # Validate and clean up each question
            cleaned = []
            for q in questions:
                cleaned.append({
                    "id": q.get("id", f"q{len(cleaned)+1}"),
                    "question": q.get("question", ""),
                    "target_function": q.get("target_function") or None,
                    "target_file": q.get("target_file") or None,
                    "topic_type": q.get("topic_type", "implementation"),
                    "expected_concepts": q.get("expected_concepts", []),
                    "scoring_rubric": {
                        "score_2": q.get("scoring_rubric", {}).get("score_2", "Trả lời đầy đủ và chính xác"),
                        "score_1": q.get("scoring_rubric", {}).get("score_1", "Trả lời một phần đúng"),
                        "score_0": q.get("scoring_rubric", {}).get("score_0", "Trả lời sai hoặc không biết"),
                    },
                    "max_score": MAX_SCORE_PER_QUESTION,
                })

            return cleaned

        except Exception as exc:
            logger.error("Question generation failed: %s", exc, exc_info=True)
            # Minimal fallback: one generic question per candidate
            fallback = []
            for i, c in enumerate(candidates, 1):
                func_names = [f.get("name", "") for f in c.get("functions", [])[:1]]
                func_str = func_names[0] if func_names else "code"
                fallback.append({
                    "id": f"q{i}",
                    "question": f"Em có thể giải thích cách em implement `{func_str}` không?",
                    "target_function": func_names[0] if func_names else None,
                    "target_file": c.get("functions", [{}])[0].get("file") if c.get("functions") else None,
                    "topic_type": c.get("type", "implementation"),
                    "expected_concepts": [],
                    "scoring_rubric": {
                        "score_2": "Giải thích rõ ràng, đầy đủ logic",
                        "score_1": "Giải thích được một phần",
                        "score_0": "Không giải thích được",
                    },
                    "max_score": MAX_SCORE_PER_QUESTION,
                })
            return fallback

    # ------------------------------------------------------------------
    # Fallback – Questions from assignment text (no code structure)
    # ------------------------------------------------------------------

    def _generate_questions_from_assignment(
        self, assignment: str, project_summary: str
    ) -> List[Dict[str, Any]]:
        """Generate exam questions purely from the assignment text.

        Used when the code analyzer found no functions (e.g. no-functions repos,
        pure-config repos, or repos whose language isn’t yet supported).
        """
        if not assignment.strip():
            return []

        prompt = f"""Bạn là giảng viên tạo câu hỏi vấn đáp dựa trên đề bài (không có code để phân tích).

=== TÓM TẮt PROJECT ===
{project_summary}

=== ĐỀ BÀI ===
{assignment[:3000]}

Tạo {MIN_QUESTIONS}–{MAX_QUESTIONS} câu hỏi vấn đáp bằng tiếng Việt.
Mỗi câu hỏi kiểm tra liệu sinh viên có hiểu rõ một yêu cầu trong đề bài không.
Không tiết lộ đáp án trong câu hỏi.
TẤT CẢ các trường (question, expected_concepts, scoring_rubric) đều phải bằng TIẾNG VIỆT.

Trả về JSON array:
[
  {{
    "id": "q1",
    "question": "...",
    "target_function": null,
    "target_file": null,
    "topic_type": "implementation",
    "expected_concepts": ["khái niệm 1 bằng tiếng Việt", "khái niệm 2"],
    "scoring_rubric": {{
      "score_2": "Tiêu chí 2 điểm",
      "score_1": "Tiêu chí 1 điểm",
      "score_0": "Tiêu chí 0 điểm"
    }},
    "max_score": 2
  }}
]

Chỉ trả về JSON array, không thêm text hay markdown."""

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON array without markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

            questions: List[Dict] = json.loads(raw)
            cleaned = []
            for q in questions:
                cleaned.append({
                    "id": q.get("id", f"q{len(cleaned)+1}"),
                    "question": q.get("question", ""),
                    "target_function": None,
                    "target_file": None,
                    "topic_type": q.get("topic_type", "implementation"),
                    "expected_concepts": q.get("expected_concepts", []),
                    "scoring_rubric": {
                        "score_2": q.get("scoring_rubric", {}).get("score_2", "Trả lời đầy đủ và chính xác"),
                        "score_1": q.get("scoring_rubric", {}).get("score_1", "Trả lời một phần đúng"),
                        "score_0": q.get("scoring_rubric", {}).get("score_0", "Trả lời sai hoặc không biết"),
                    },
                    "max_score": MAX_SCORE_PER_QUESTION,
                })
            return cleaned

        except Exception as exc:
            logger.error("Assignment-based question generation failed: %s", exc, exc_info=True)
            return []
