"""FastAPI application for Teacher Chatbot API."""
import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from src.models.schemas import HealthResponse
from src.api import chat_router, upload_router, github_router, qa_router, analyze_router
from src.api.project_qa_v2 import router as project_qa_v2_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Teacher Chatbot API",
    description="AI-powered teacher chatbot with RAG capabilities and oral exam assessment",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(github_router)
app.include_router(analyze_router)
app.include_router(qa_router)
app.include_router(project_qa_v2_router)

# Root endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API info."""
    return {
        "status": "online",
        "message": "Teacher Chatbot API v2.0 is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Verify environment variables
        required_vars = ["OPENAI_API_KEY", "ZILLIZ_CLOUD_URI", "ZILLIZ_CLOUD_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            return {
                "status": "unhealthy",
                "message": f"Missing environment variables: {', '.join(missing)}"
            }
        
        return {
            "status": "healthy",
            "message": "All systems operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": str(e)
        }



@app.get("/test")
async def test_page():
    """Serve the SSE streaming test UI."""
    return FileResponse(
        Path(__file__).parent / "test_sse.html",
        media_type="text/html",
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    try:
        logger.info("🚀 Initializing Teacher Chatbot API v3.0...")
        logger.info("✅ API initialized successfully!")
        logger.info("📚 Features:")
        logger.info("   • /analyze/project  → analysis + question_bank + project_summary")
        logger.info("   • /qa/start         → oral exam greeting + first question")
        logger.info("   • /qa/answer        → per-turn evaluation with score locking")
        logger.info("   • /upload, /github, /chat still available")
    except Exception as e:
        logger.warning(f"⚠️ Warning during initialization: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
