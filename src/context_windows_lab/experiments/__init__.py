"""
Experiments package for Context Windows Lab.

Contains implementations of all four experiments:
1. Needle in Haystack (Lost in the Middle)
2. Context Window Size Impact
3. RAG vs Full Context
4. Context Engineering Strategies
"""

from .base import BaseExperiment, ExperimentResult
from .needle_haystack import NeedleInHaystackExperiment
from .context_size import ContextSizeExperiment
from .rag_impact import RAGImpactExperiment
from .context_engineering import ContextEngineeringExperiment

__all__ = [
    "BaseExperiment",
    "ExperimentResult",
    "NeedleInHaystackExperiment",
    "ContextSizeExperiment",
    "RAGImpactExperiment",
    "ContextEngineeringExperiment",
]
