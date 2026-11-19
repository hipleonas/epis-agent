"""Analyze API endpoint — Generate oral-exam question bank from a code submission."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal
import logging

from src.services.analyze_agent import AnalyzeAgent
from src.services.code_analyzer import analyze_code_structure
from src.services.question_generator import QuestionGeneratorService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""
    repository_structure: Dict[str, Any]
    assignment_requirements: str
    llm_provider: Literal["open_ai", "self_host"] = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )


class AnalyzeResponse(BaseModel):
    """Response from /analyze/project.

    Store all three fields in project_submission (PostgreSQL):
      - question_bank   → JSONB column  project_submission.question_bank
      - project_summary → TEXT  column  project_submission.project_summary
      - total_questions → INT   column  project_submission.total_questions
        (or derive it as len(question_bank) — whichever is easier in your ORM)
    """
    question_bank: List[Dict[str, Any]]
    project_summary: str
    total_questions: int


@router.post("/project", response_model=AnalyzeResponse)
async def analyze_project(request: AnalyzeRequest):
    """Analyse a student's code submission and produce a pre-baked oral-exam question bank.

    Request body:
        repository_structure    – Full repo tree with file contents (from /github/fetch-repo)
        assignment_requirements – Plain-text assignment description

    Response:
        question_bank   – 5–7 pre-generated oral exam questions with rubrics.
                          Store in project_submission.question_bank (JSONB).
        project_summary – Short (~150-word) plain-text project summary.
                          Store in project_submission.project_summary (TEXT).
        total_questions – Number of questions generated.
                          Store in project_submission.total_questions (INT).

    Internally this endpoint runs three steps:
      1. Extract functions/classes from the repository code.
      2. Map functions to assignment requirements (AnalyzeAgent).
      3. Generate question bank + summary (QuestionGeneratorService).
    Only steps 2 and 3 make external LLM calls; step 1 is purely local.
    """
    try:
        logger.info("📥 /analyze/project: request received")

        repo_structure = request.repository_structure
        logger.info(f"📂 Repository: {repo_structure.get('name', 'Unknown')}")

        # Step 1: Extract functions from the repository (local, no LLM)
        _, functions_list = analyze_code_structure(repo_structure)
        logger.info(f"🔍 Found {len(functions_list)} functions")

        if not functions_list:
            logger.warning(
                "No functions extracted from repository. "
                "Proceeding without function list — question bank will be based on project structure only."
            )

        # Step 2: Map requirements → functions (LLM call 1)
        analyzer = AnalyzeAgent(llm_provider=request.llm_provider)
        analysis = analyzer.analyze(
            assignment=request.assignment_requirements,
            functions_list=functions_list,
            repository_structure=repo_structure,
        )
        logger.info(f"✅ Analysis: {analysis['metadata']['total_requirements']} requirements mapped")

        # Step 3: Generate question bank + project summary (LLM call 2)
        generator = QuestionGeneratorService(llm_provider=request.llm_provider)
        qa_data = generator.generate(analysis, assignment=request.assignment_requirements)

        question_bank = qa_data["question_bank"]
        project_summary = qa_data["project_summary"]
        total_questions = len(question_bank)
        logger.info(f"✅ Question bank: {total_questions} questions generated")

        return AnalyzeResponse(
            question_bank=question_bank,
            project_summary=project_summary,
            total_questions=total_questions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ /analyze/project failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích: {str(exc)}")

