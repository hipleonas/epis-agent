"""Embedding utilities using OpenAI."""
import os
import logging
import time
from typing import List
from openai import OpenAI

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self, model: str = None):
        """Initialize the embedding service.
        
        Args:
            model: OpenAI embedding model name. Defaults to env variable.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        logger.info(f"⏳ Calling OpenAI API to generate {len(texts)} embeddings (model: {self.model})...")
        api_start = time.time()
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        api_time = time.time() - api_start
        embeddings = [item.embedding for item in response.data]
        logger.info(f"✓ Received {len(embeddings)} embeddings from OpenAI in {api_time:.2f}s")
        
        return embeddings
