"""
Context Windows Lab - LLM Experiments Package

A comprehensive toolkit for experimenting with LLM context windows,
demonstrating key phenomena like "Lost in the Middle" and comparing
RAG strategies with full-context approaches.

Team: OmerAndYogever
Course: LLMs and Multi-Agent Systems
"""

__version__ = "1.0.0"
__author__ = "Omer & Yogev Cuperman"
__team__ = "OmerAndYogever"

from .config import Config
from .experiments import (
    NeedleInHaystackExperiment,
    ContextSizeExperiment,
    RAGImpactExperiment,
    ContextEngineeringExperiment,
)

__all__ = [
    "Config",
    "NeedleInHaystackExperiment",
    "ContextSizeExperiment",
    "RAGImpactExperiment",
    "ContextEngineeringExperiment",
]
