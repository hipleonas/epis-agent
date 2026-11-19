"""Azure Blob Storage utilities."""
import logging
import tempfile
import os
import requests
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AzureBlobStorage:
    """Utilities for working with Azure Blob Storage."""
    
    @staticmethod
    def download_from_url(url: str, save_dir: Optional[str] = None) -> Tuple[str, str]:
        """Download file from Azure Blob Storage URL.
        
        Args:
            url: Azure blob storage URL (publicly accessible or with SAS token).
            save_dir: Directory to save file. Uses temp dir if None.
            
        Returns:
            Tuple of (file_path, filename).
            
        Raises:
            Exception: If download fails.
        """
        logger.info(f"⏳ Downloading from Azure Blob Storage...")
        logger.info(f"   URL: {url[:80]}...")
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            filename = "downloaded_file"
        
        logger.info(f"   📄 Filename: {filename}")
        
        # Download file
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Determine file type from content-type header if available
            content_type = response.headers.get('content-type', '').lower()
            
            # Get file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                # Try to determine from content-type
                if 'pdf' in content_type:
                    file_ext = '.pdf'
                elif 'word' in content_type or 'officedocument' in content_type:
                    file_ext = '.docx'
                elif 'text' in content_type:
                    file_ext = '.txt'
                else:
                    file_ext = ''
                
                filename = filename + file_ext
            
            # Save to file
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, filename)
            else:
                # Use temp directory
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=file_ext,
                    dir=tempfile.gettempdir()
                )
                file_path = temp_file.name
                temp_file.close()
            
            # Write content
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✓ Downloaded successfully: {file_size_mb:.2f} MB")
            logger.info(f"   Saved to: {file_path}")
            
            return file_path, filename
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Download failed: {str(e)}")
            raise Exception(f"Failed to download from Azure: {str(e)}")
    
    @staticmethod
    def detect_file_type(url: str, file_path: Optional[str] = None) -> str:
        """Detect file type from URL or file path.
        
        Args:
            url: Azure blob storage URL.
            file_path: Local file path (optional).
            
        Returns:
            File extension (pdf, docx, txt).
        """
        # Try from URL path first
        parsed_url = urlparse(url)
        url_path = parsed_url.path.lower()
        
        if url_path.endswith('.pdf'):
            return 'pdf'
        elif url_path.endswith('.docx') or url_path.endswith('.doc'):
            return 'docx'
        elif url_path.endswith('.txt'):
            return 'txt'
        
        # Try from file path
        if file_path:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext in ['pdf', 'docx', 'doc', 'txt']:
                return ext if ext != 'doc' else 'docx'
        
        # Default to txt if can't determine
        logger.warning(f"⚠️ Could not detect file type, defaulting to txt")
        return 'txt'
