"""
Token Counter utility for Context Windows Lab.

Provides accurate token counting for Claude models using tiktoken.
"""

from typing import Optional

import tiktoken


class TokenCounter:
    """
    Counts tokens for text using Claude-compatible tokenization.
    
    Uses cl100k_base encoding which is similar to Claude's tokenizer.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token counter.
        
        Args:
            encoding_name: The tiktoken encoding to use
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_documents(self, documents: list[str]) -> dict[str, int]:
        """
        Count tokens for a list of documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Dictionary with per-document and total token counts
        """
        counts = []
        for doc in documents:
            counts.append(self.count(doc))
        
        return {
            "per_document": counts,
            "total": sum(counts),
            "average": sum(counts) / len(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "min": min(counts) if counts else 0,
        }
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to a maximum number of tokens.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def split_by_tokens(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int = 0
    ) -> list[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        
        Args:
            text: The text to split
            chunk_size: Target size of each chunk in tokens
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start = end - overlap if overlap > 0 else end
        
        return chunks


# Singleton instance for convenience
_default_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get the default token counter instance."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens in text."""
    return get_token_counter().count(text)
