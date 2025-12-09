"""
Configuration management for Context Windows Lab.

Handles environment variables, default settings, and experiment parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """
    Central configuration for all experiments.
    
    Loads settings from environment variables with sensible defaults.
    Supports both API-based (Claude) and local (Ollama) LLM backends.
    """
    
    # API Configuration
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    
    # Experiment Settings
    num_runs: int = 5
    random_seed: int = 42
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    chroma_persist_dir: Path = field(default_factory=lambda: Path("./data/chromadb"))
    
    # Document Generation
    words_per_document: int = 200
    default_num_documents: int = 5
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 3
    
    def __post_init__(self):
        """Load environment variables and ensure directories exist."""
        load_dotenv()
        
        # Load from environment
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.claude_model = os.getenv("CLAUDE_MODEL", self.claude_model)
        self.max_tokens = int(os.getenv("MAX_TOKENS", self.max_tokens))
        self.num_runs = int(os.getenv("NUM_EXPERIMENT_RUNS", self.num_runs))
        self.random_seed = int(os.getenv("RANDOM_SEED", self.random_seed))
        
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR")
        if chroma_dir:
            self.chroma_persist_dir = Path(chroma_dir)
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def has_api_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your-anthropic-api-key-here")
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.has_api_key:
            errors.append("ANTHROPIC_API_KEY not set in environment")
        
        if self.max_tokens < 100:
            errors.append(f"max_tokens too low: {self.max_tokens}")
        
        if self.num_runs < 1:
            errors.append(f"num_runs must be at least 1: {self.num_runs}")
        
        return errors
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()
    
    def __repr__(self) -> str:
        return (
            f"Config(model={self.claude_model}, "
            f"max_tokens={self.max_tokens}, "
            f"num_runs={self.num_runs}, "
            f"has_api_key={self.has_api_key})"
        )
