"""Pydantic models for API requests and responses."""
from typing import Optional, List, Literal, Dict, Any, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

# Valid provider identifiers — used across all request schemas that trigger LLM calls.
LLMProvider = Literal["open_ai", "self_host"]

# ---------------------------------------------------------------------------
# Oral Exam Q&A Models
# ---------------------------------------------------------------------------

class ScoringRubric(BaseModel):
    """Rubric describing how to award 0 / 1 / 2 points for a question."""
    score_2: str = Field(..., description="Criteria for full marks (2 pts)")
    score_1: str = Field(..., description="Criteria for partial marks (1 pt)")
    score_0: str = Field(..., description="Criteria for zero marks (0 pts)")


class QuestionBankItem(BaseModel):
    """A single pre-generated oral exam question."""
    id: str = Field(..., description="Unique question ID, e.g. 'q1'")
    question: str = Field(..., description="The question text in Vietnamese")
    target_function: Optional[str] = Field(None, description="Primary function being asked about")
    target_file: Optional[str] = Field(None, description="File where target_function lives")
    topic_type: Literal["inappropriate_code", "implementation", "design_decision", "update_suggestion"] = Field(
        ..., description="Category of the question"
    )
    expected_concepts: List[str] = Field(default_factory=list, description="Key concepts a good answer should cover")
    scoring_rubric: ScoringRubric = Field(..., description="Rubric for awarding 0 / 1 / 2 points")
    max_score: int = Field(2, description="Maximum points for this question")


class QAStartRequest(BaseModel):
    """Request to start a new oral exam session (get greeting + first question)."""
    question_bank: List[QuestionBankItem] = Field(..., description="All pre-generated questions for this session")
    project_summary: str = Field(..., description="Short (~150 word) human-readable project summary")
    max_questions: int = Field(
        5,
        ge=1,
        description="Maximum number of questions to ask in this session. FastAPI will truncate the question bank to this limit. Defaults to 5.",
    )
    llm_provider: LLMProvider = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )



class QAStartResponse(BaseModel):
    """Response from starting the oral exam session."""
    response: str = Field(..., description="Greeting message + first question text")


class ScoreLockedInfo(BaseModel):
    """Already-locked score for the current question (sent by backend on follow-up turns)."""
    question_id: str
    score: int
    max_score: int


class QAAnswerRequest(BaseModel):
    """Request for each student answer turn during the oral exam."""
    user_message: str = Field(..., description="Student's answer or follow-up message")
    current_question: QuestionBankItem = Field(..., description="The question currently being discussed")
    current_question_turns: int = Field(
        ...,
        description=(
            "How many student turns have occurred on this question so far. "
            "0 = this is the FIRST answer (grade will be locked). "
            "1+ = follow-up turns (grade already locked)."
        )
    )
    score_locked: Optional[ScoreLockedInfo] = Field(
        None,
        description="Already-locked score if current_question_turns > 0, else null"
    )
    is_last_question: bool = Field(
        False,
        description="True when this is the final question in the session"
    )
    project_summary: str = Field(..., description="Short project summary for agent context")
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent conversation messages [{role, content}] — NO full repo JSON, last ~10 messages only"
    )
    llm_provider: LLMProvider = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )


class ScoringEvent(BaseModel):
    """Score awarded for a question (only present on the first-answer turn)."""
    question_id: str
    score: int = Field(..., description="0, 1, or 2")
    max_score: int = Field(2)


class QAAnswerResponse(BaseModel):
    """Response from the oral exam agent for one student turn."""
    response: str = Field(..., description="Teacher's natural-language response")
    scoring_event: Optional[ScoringEvent] = Field(
        None,
        description="Present only on the first-answer turn; null on follow-up turns"
    )
    advance_to_next: bool = Field(
        False,
        description="True when the agent has finished discussing this question and is ready for the next"
    )
    is_complete: bool = Field(
        False,
        description="True when this was the last question and the entire session is complete"
    )


class HistoryMessage(BaseModel):
    """Conversation history message."""
    index: int = Field(..., description="Message order in conversation")
    role: Literal["user", "assistant"] = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request payload."""
    query: str = Field(..., description="User's question or message")
    course_id: Optional[str] = Field(None, description="Current course ID (optional)")
    module_id: Optional[str] = Field(None, description="Current module ID (optional)")
    history: Optional[List[HistoryMessage]] = Field(default=None, description="Conversation history (optional)")
    llm_provider: LLMProvider = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )


class SourceInfo(BaseModel):
    """Source document information."""
    id: str
    score: float
    text: str
    course_id: Optional[str]
    module_id: Optional[str]
    from_src: Optional[str]


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str = Field(..., description="Bot's response")
    intent: str = Field(..., description="Detected intent (2-valued, backwards compatible): 'chat' or 'learning'")
    sources: List[SourceInfo] = Field(default_factory=list, description="Retrieved source documents (for RAG)")
    label: str = Field(
        "learning",
        description="4-way retrieve-first classification: 'chat' | 'learning' | 'cross_module' | 'out_of_corpus'"
    )
    refused: bool = Field(False, description="True when the answer is a canned refusal (cross_module/out_of_corpus)")
    refusal_reason: Optional[Literal["cross_module", "out_of_corpus"]] = Field(
        None, description="Why the question was refused, if it was"
    )
    suggested_module_id: Optional[str] = Field(
        None, description="Module likely to contain the answer, only set when refusal_reason='cross_module'"
    )


class SocraticDocument(BaseModel):
    """Document context for Socratic session initialization."""
    type: Literal["markdown", "pdf"] = Field(..., description="Document type")
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Extracted plain text content")
    file_path: Optional[str] = Field(None, description="Optional local file path to read and extract content on server")


class SocraticInitRequest(BaseModel):
    """Initialize Socratic session state once when user enters project page."""
    session_id: str = Field(..., description="Session ID from frontend")
    problem: Optional[str] = Field(None, description="Prebuilt problem context text")
    documents: List[SocraticDocument] = Field(
        default_factory=list,
        description="Optional markdown/pdf documents to build problem context"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    reset_history: bool = Field(True, description="Reset session history on init")


class SocraticInitResponse(BaseModel):
    """Response after Socratic session init."""
    session_id: str
    initialized: bool
    has_problem: bool
    problem_length: int


class SocraticChatRequest(BaseModel):
    """Stateful Socratic chat request using existing session state."""
    session_id: str = Field(..., description="Session ID that was initialized before")
    query: str = Field(..., description="User message")
    llm_provider: LLMProvider = Field(
        "open_ai",
        description="LLM provider to use: 'open_ai' (default, OpenAI cloud) or 'self_host' (vLLM)",
    )


class SocraticChatResponse(BaseModel):
    """Stateful Socratic chat response."""
    session_id: str
    response: str
    intent: str
    history_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class FileUploadResult(BaseModel):
    """Result for a single file upload."""
    filename: str
    success: bool
    chunks_count: int = 0
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response for file upload endpoints."""
    total_files: int
    successful: int
    failed: int
    total_chunks: int
    course_id: str
    module_id: str
    results: List[FileUploadResult]


class AzureUploadRequest(BaseModel):
    """Request for uploading from Azure Blob Storage URLs."""
    urls: List[str] = Field(..., description="List of Azure blob storage URLs")
    course_id: str = Field(..., description="Course ID")
    module_id: str = Field(..., description="Module ID")
    from_src: Optional[str] = Field(None, description="Source name (optional, uses filename if not provided)")


# GitHub Repository Models

class GitHubRepoRequest(BaseModel):
    """Request for fetching GitHub repository."""
    repo_url: str = Field(..., description="GitHub repository URL (e.g., https://github.com/owner/repo)")


class GitHubFileNode(BaseModel):
    """Single file or folder node in repository tree."""
    name: str = Field(..., description="File or folder name")
    type: Literal["file", "folder"] = Field(..., description="Node type")
    extension: Optional[str] = Field(None, description="File extension (e.g., '.py') - only for files")
    content: Optional[str] = Field(None, description="File content - null for folders, binary files, or large files")
    size: Optional[int] = Field(None, description="File size in bytes - only for files")
    is_binary: Optional[bool] = Field(None, description="Whether file is binary (image, video, pdf, etc.)")
    reason: Optional[str] = Field(None, description="Reason why content is null (if applicable)")
    children: Optional[List['GitHubFileNode']] = Field(default_factory=list, description="Child nodes - only for folders")


class GitHubRepoResponse(BaseModel):
    """Response with complete repository structure."""
    name: str = Field(..., description="Repository name")
    type: Literal["folder"] = Field("folder", description="Root is always a folder")
    children: List[GitHubFileNode] = Field(..., description="Repository contents")
    total_files: int = Field(..., description="Total number of files")
    total_folders: int = Field(..., description="Total number of folders")
    skipped_files: int = Field(..., description="Number of files skipped (binary, too large, etc.)")


# Update forward references for recursive model
GitHubFileNode.model_rebuild()

