"""Text chunking utilities using LangChain."""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    
    This uses LangChain's optimized text splitting which intelligently splits on:
    - Paragraphs (\n\n)
    - Lines (\n)
    - Sentences (. ! ?)
    - Words (spaces)
    
    Args:
        text: Input text to chunk.
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    # Use LangChain's RecursiveCharacterTextSplitter for optimal chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
