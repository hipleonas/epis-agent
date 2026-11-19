"""GitHub repository fetcher - fetch repository structure and file contents."""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Binary file extensions to exclude from content reading
BINARY_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico', '.webp',
    # Videos
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm',
    # Audio
    '.mp3', '.wav', '.flac', '.aac', '.ogg',
    # Archives
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
    # Documents
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Executables
    '.exe', '.dll', '.so', '.dylib', '.bin',
    # Other binary
    '.pyc', '.pyo', '.class', '.o', '.a'
}

# Patterns to exclude from tree
EXCLUDE_PATTERNS = [
    '.git',
    'node_modules',
    '__pycache__',
    '.pytest_cache',
    '.venv',
    'venv',
    'env',
    '.env',
    '.idea',
    '.vscode',
    'dist',
    'build',
    '*.egg-info',
    '.DS_Store',
    'Thumbs.db'
]

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB in bytes


class GitHubFetcher:
    """Fetch GitHub repository structure and contents."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub fetcher.
        
        Args:
            token: GitHub personal access token (optional but recommended).
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        self.headers["Accept"] = "application/vnd.github.v3+json"
    
    def parse_github_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repo name.
        
        Args:
            url: GitHub repository URL.
            
        Returns:
            Tuple of (owner, repo_name).
            
        Raises:
            ValueError: If URL is not a valid GitHub repository URL.
        """
        # Clean the URL
        url = url.strip().rstrip('/')
        
        # Match various GitHub URL formats
        patterns = [
            r'github\.com/([^/]+)/([^/]+?)(?:\.git)?$',  # https://github.com/owner/repo or .git
            r'github\.com/([^/]+)/([^/]+)',  # https://github.com/owner/repo/...
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                owner, repo = match.groups()
                # Remove .git suffix if present
                repo = repo.replace('.git', '')
                return owner, repo
        
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    def should_exclude(self, path: str) -> bool:
        """Check if path should be excluded based on patterns.
        
        Args:
            path: File or folder path.
            
        Returns:
            True if should be excluded, False otherwise.
        """
        path_lower = path.lower()
        
        for pattern in EXCLUDE_PATTERNS:
            if pattern.startswith('*'):
                # Wildcard pattern
                if path_lower.endswith(pattern[1:]):
                    return True
            else:
                # Exact match or path contains pattern
                if pattern in path_lower or path_lower.startswith(pattern + '/'):
                    return True
        
        return False
    
    def is_binary_file(self, filename: str) -> bool:
        """Check if file is binary based on extension.
        
        Args:
            filename: Name of the file.
            
        Returns:
            True if file is binary, False otherwise.
        """
        _, ext = os.path.splitext(filename.lower())
        return ext in BINARY_EXTENSIONS
    
    def get_file_extension(self, filename: str) -> str:
        """Get file extension.
        
        Args:
            filename: Name of the file.
            
        Returns:
            File extension including the dot (e.g., '.py') or empty string.
        """
        _, ext = os.path.splitext(filename)
        return ext
    
    def fetch_repo_info(self, owner: str, repo: str) -> Dict:
        """Fetch basic repository information.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            
        Returns:
            Repository information including default branch.
            
        Raises:
            requests.HTTPError: If API request fails.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def fetch_tree(self, owner: str, repo: str, branch: str = "main") -> List[Dict]:
        """Fetch complete repository tree structure from GitHub API.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            branch: Branch name (default: main).
            
        Returns:
            List of tree items with path, type, size, and sha.
            
        Raises:
            requests.HTTPError: If API request fails.
        """
        # First get the commit SHA for the branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        
        logger.info(f"Fetching tree for {owner}/{repo} (branch: {branch})")
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        tree_items = data.get("tree", [])
        
        logger.info(f"Found {len(tree_items)} items in repository tree")
        return tree_items
    
    def fetch_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str = "main"
    ) -> Optional[str]:
        """Fetch file content from GitHub raw URL.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path in repository.
            branch: Branch name.
            
        Returns:
            File content as string, or None if failed.
        """
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Try to decode as UTF-8 text
            try:
                return response.content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode {path} as UTF-8, treating as binary")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch content for {path}: {e}")
            return None
    
    def build_tree_structure(
        self,
        tree_items: List[Dict],
        owner: str,
        repo: str,
        branch: str = "main"
    ) -> Dict:
        """Build nested tree structure with file contents.
        
        Args:
            tree_items: List of tree items from GitHub API.
            owner: Repository owner.
            repo: Repository name.
            branch: Branch name.
            
        Returns:
            Nested tree structure as dictionary.
        """
        # Statistics
        stats = {
            "total_files": 0,
            "total_folders": 0,
            "skipped_files": 0
        }
        
        # Build tree structure
        root = {
            "name": repo,
            "type": "folder",
            "children": []
        }
        
        # Create a mapping of paths to nodes for easy lookup
        path_map = {"": root}
        
        # Sort items by path to ensure parents are created before children
        sorted_items = sorted(tree_items, key=lambda x: x["path"])
        
        for item in sorted_items:
            path = item["path"]
            item_type = item["type"]  # 'blob' for file, 'tree' for folder
            size = item.get("size", 0)
            
            # Skip excluded paths
            if self.should_exclude(path):
                logger.debug(f"Excluding: {path}")
                continue
            
            # Get parent path and node name
            path_parts = path.split('/')
            node_name = path_parts[-1]
            parent_path = '/'.join(path_parts[:-1])
            
            # Get or create parent node
            if parent_path not in path_map:
                # This shouldn't happen if sorted correctly, but handle it
                continue
            
            parent_node = path_map[parent_path]
            
            if item_type == "tree":
                # Folder
                folder_node = {
                    "name": node_name,
                    "type": "folder",
                    "children": []
                }
                parent_node["children"].append(folder_node)
                path_map[path] = folder_node
                stats["total_folders"] += 1
                
            elif item_type == "blob":
                # File
                stats["total_files"] += 1
                
                file_node = {
                    "name": node_name,
                    "type": "file",
                    "extension": self.get_file_extension(node_name),
                    "size": size
                }
                
                # Check if file should have content
                is_binary = self.is_binary_file(node_name)
                is_too_large = size > MAX_FILE_SIZE
                
                if is_binary:
                    file_node["content"] = None
                    file_node["is_binary"] = True
                    file_node["reason"] = "Binary file"
                    stats["skipped_files"] += 1
                    
                elif is_too_large:
                    file_node["content"] = None
                    file_node["reason"] = f"File too large ({size} bytes > {MAX_FILE_SIZE} bytes)"
                    stats["skipped_files"] += 1
                    
                else:
                    # Fetch content
                    logger.debug(f"Fetching content for: {path}")
                    content = self.fetch_file_content(owner, repo, path, branch)
                    
                    if content is not None:
                        file_node["content"] = content
                    else:
                        file_node["content"] = None
                        file_node["reason"] = "Failed to fetch or decode content"
                        stats["skipped_files"] += 1
                
                parent_node["children"].append(file_node)
        
        # Add statistics to root
        root["total_files"] = stats["total_files"]
        root["total_folders"] = stats["total_folders"]
        root["skipped_files"] = stats["skipped_files"]
        
        return root
    
    def fetch_repository(self, repo_url: str) -> Dict:
        """Fetch complete repository structure and contents.
        
        Args:
            repo_url: GitHub repository URL.
            
        Returns:
            Complete tree structure with file contents.
            
        Raises:
            ValueError: If URL is invalid.
            requests.HTTPError: If GitHub API request fails.
        """
        # Parse URL
        owner, repo = self.parse_github_url(repo_url)
        logger.info(f"Fetching repository: {owner}/{repo}")
        
        # Get repository info (including default branch)
        repo_info = self.fetch_repo_info(owner, repo)
        default_branch = repo_info.get("default_branch", "main")
        logger.info(f"Default branch: {default_branch}")
        
        # Fetch tree structure
        tree_items = self.fetch_tree(owner, repo, default_branch)
        
        # Build nested structure with content
        tree_structure = self.build_tree_structure(tree_items, owner, repo, default_branch)
        
        logger.info(f"✅ Successfully fetched repository structure")
        return tree_structure
