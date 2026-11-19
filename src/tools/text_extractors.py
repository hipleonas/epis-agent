"""Text extraction tools for various file formats."""
import logging
import tempfile
import os
from typing import Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from various file formats."""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file using LangChain PyPDFLoader.
        
        Args:
            file_path: Path to PDF file.
            
        Returns:
            Extracted text content.
        """
        logger.info(f"⏳ Extracting text from PDF: {file_path}")
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        logger.info(f"   📄 PDF has {len(pages)} pages")
        
        text_content = ""
        for page_num, page in enumerate(pages, 1):
            text_content += page.page_content + "\n\n"
            if page_num % 10 == 0:
                logger.info(f"   ⏳ Processed {page_num}/{len(pages)} pages...")
        
        logger.info(f"✓ PDF extraction complete: {len(text_content):,} characters")
        return text_content
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file using LangChain Docx2txtLoader.
        
        Args:
            file_path: Path to DOCX file.
            
        Returns:
            Extracted text content.
        """
        logger.info(f"⏳ Extracting text from DOCX: {file_path}")
        
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        logger.info(f"✓ DOCX extraction complete: {len(text_content):,} characters")
        return text_content
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file.
        
        Args:
            file_path: Path to TXT file.
            
        Returns:
            Extracted text content.
        """
        logger.info(f"⏳ Extracting text from TXT: {file_path}")
        
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        logger.info(f"✓ TXT extraction complete: {len(text_content):,} characters")
        return text_content
    
    @staticmethod
    def extract_text(file_path: str, file_type: Optional[str] = None) -> str:
        """Extract text from file based on file type.
        
        Args:
            file_path: Path to file.
            file_type: File extension (pdf, docx, txt). Auto-detected if None.
            
        Returns:
            Extracted text content.
            
        Raises:
            ValueError: If file type is not supported.
        """
        # Auto-detect file type from extension
        if not file_type:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return TextExtractor.extract_from_pdf(file_path)
        elif file_type in ['docx', 'doc']:
            return TextExtractor.extract_from_docx(file_path)
        elif file_type in ['txt', 'text']:
            return TextExtractor.extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported: pdf, docx, txt")
