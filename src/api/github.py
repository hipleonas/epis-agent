"""GitHub repository API endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from src.models.schemas import GitHubRepoRequest, GitHubRepoResponse
from src.tools.github_fetcher import GitHubFetcher
import requests

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/github", tags=["github"])


@router.post("/fetch-repo", response_model=GitHubRepoResponse)
async def fetch_repository(request: GitHubRepoRequest):
    """
    Fetch GitHub repository structure and file contents.
    
    This endpoint fetches the complete tree structure of a public GitHub repository
    and includes the content of text files (up to 1MB). Binary files and excluded
    patterns are included in the tree but without content.
    
    **Request Body:**
    ```json
    {
      "repo_url": "https://github.com/owner/repository"
    }
    ```
    
    **Example URLs:**
    - `https://github.com/fastapi/fastapi`
    - `https://github.com/python/cpython`
    - `https://github.com/owner/repo.git`
    
    **Response includes:**
    - Complete nested folder/file structure
    - File contents (for text files < 1MB)
    - File extensions, sizes, and metadata
    - Statistics (total files, folders, skipped files)
    
    **Files excluded:**
    - Binary files (images, videos, PDFs, executables, etc.)
    - Files larger than 1MB
    - Common ignore patterns (.git/, node_modules/, __pycache__, .env, etc.)
    
    **Rate Limits:**
    - Without GitHub token: 60 requests/hour
    - With GitHub token: 5,000 requests/hour
    - Set GITHUB_TOKEN in .env file for higher limits
    
    **Error Responses:**
    - `400`: Invalid GitHub URL format
    - `404`: Repository not found or is private
    - `403`: Rate limit exceeded or authentication required
    - `500`: Server error during fetching
    """
    try:
        logger.info(f"Fetching repository: {request.repo_url}")
        
        # Initialize fetcher
        fetcher = GitHubFetcher()
        
        # Fetch repository structure and contents
        tree_structure = fetcher.fetch_repository(request.repo_url)
        
        logger.info(f"✅ Successfully fetched repository structure")
        return tree_structure
        
    except ValueError as e:
        # Invalid URL format
        logger.error(f"Invalid URL: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid GitHub URL",
                "message": str(e),
                "example": "https://github.com/owner/repository"
            }
        )
        
    except requests.HTTPError as e:
        # GitHub API errors
        status_code = e.response.status_code
        
        if status_code == 404:
            logger.error(f"Repository not found: {request.repo_url}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Repository not found",
                    "message": "The repository does not exist or is private. Ensure the URL is correct and the repository is public.",
                    "repo_url": request.repo_url
                }
            )
        elif status_code == 403:
            logger.error(f"Rate limit or permission error: {e}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access forbidden",
                    "message": "GitHub API rate limit exceeded or authentication required. Set GITHUB_TOKEN in .env file for higher rate limits.",
                    "hint": "Get a token at https://github.com/settings/tokens"
                }
            )
        else:
            logger.error(f"GitHub API error: {e}")
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": "GitHub API error",
                    "message": str(e)
                }
            )
            
    except requests.RequestException as e:
        # Network or connection errors
        logger.error(f"Network error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Network error",
                "message": "Failed to connect to GitHub. Please check your internet connection.",
                "details": str(e)
            }
        )
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error fetching repository: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred while fetching the repository.",
                "details": str(e)
            }
        )


@router.get("/health")
async def health_check():
    """
    Health check for GitHub API endpoints.
    
    Returns service status and configuration info.
    """
    import os
    
    has_token = bool(os.getenv("GITHUB_TOKEN"))
    
    return {
        "status": "healthy",
        "service": "GitHub Repository Fetcher",
        "github_token_configured": has_token,
        "rate_limit": "5,000/hour" if has_token else "60/hour (unauthenticated)",
        "recommendation": "Set GITHUB_TOKEN in .env for higher rate limits" if not has_token else "Token configured ✓"
    }
