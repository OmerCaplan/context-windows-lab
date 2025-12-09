"""
LLM Client wrapper for Context Windows Lab.

Provides a unified interface for querying Claude models,
with support for token counting and response evaluation.
"""

import time
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic

from ..config import Config


@dataclass
class LLMResponse:
    """
    Response from an LLM query.
    
    Attributes:
        content: The text response from the model
        latency: Time taken to get the response (seconds)
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        model: The model used for the query
    """
    content: str
    latency: float
    input_tokens: int
    output_tokens: int
    model: str
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in the request."""
        return self.input_tokens + self.output_tokens


class LLMClient:
    """
    Unified client for LLM queries.
    
    Wraps the Anthropic API with convenience methods for:
    - Simple queries with context
    - Token counting
    - Response timing
    - Error handling
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config.from_env()
        
        if self.config.has_api_key:
            self.client = Anthropic(api_key=self.config.anthropic_api_key)
        else:
            self.client = None
    
    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Query the LLM with context and a question.
        
        Args:
            context: The context/documents to provide
            question: The question to ask about the context
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (uses config default if None)
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            RuntimeError: If API key is not configured
        """
        if not self.client:
            raise RuntimeError(
                "LLM client not initialized. Please set ANTHROPIC_API_KEY environment variable."
            )
        
        max_tokens = max_tokens or self.config.max_tokens
        
        # Build the message
        user_message = f"""Context:
{context}

Question: {question}"""
        
        messages = [{"role": "user", "content": user_message}]
        
        # Time the request
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant. Answer questions based on the provided context.",
            messages=messages,
        )
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=response.content[0].text,
            latency=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.config.claude_model,
        )
    
    def query_simple(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Simple query without separate context.
        
        Args:
            prompt: The full prompt to send
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse with content and metadata
        """
        if not self.client:
            raise RuntimeError(
                "LLM client not initialized. Please set ANTHROPIC_API_KEY environment variable."
            )
        
        max_tokens = max_tokens or self.config.max_tokens
        
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=response.content[0].text,
            latency=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.config.claude_model,
        )
    
    def evaluate_response(
        self,
        response: str,
        expected_answer: str,
        strict: bool = False,
    ) -> tuple[bool, float]:
        """
        Evaluate if a response contains the expected answer.
        
        Args:
            response: The LLM's response
            expected_answer: The expected answer to find
            strict: If True, requires exact match; if False, checks containment
            
        Returns:
            Tuple of (is_correct, confidence_score)
        """
        response_lower = response.lower().strip()
        expected_lower = expected_answer.lower().strip()
        
        if strict:
            is_correct = response_lower == expected_lower
            confidence = 1.0 if is_correct else 0.0
        else:
            is_correct = expected_lower in response_lower
            # Calculate a simple confidence based on how prominent the answer is
            if is_correct:
                # Higher confidence if answer is a larger portion of response
                confidence = min(1.0, len(expected_lower) / max(len(response_lower), 1) * 2)
            else:
                confidence = 0.0
        
        return is_correct, confidence
    
    def is_available(self) -> bool:
        """Check if the LLM client is properly configured and available."""
        return self.client is not None
