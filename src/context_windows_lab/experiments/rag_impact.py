"""
Experiment 3: RAG Impact

Compares two retrieval strategies:
1. Full Context: All documents in the context window
2. RAG: Only relevant documents retrieved via similarity search

Expected: RAG provides better accuracy and faster responses.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseExperiment
from ..utils.document_generator import DocumentGenerator
from ..utils.token_counter import TokenCounter


class RAGImpactExperiment(BaseExperiment):
    """
    Compares RAG (Retrieval Augmented Generation) vs Full Context approaches.
    
    Tests accuracy, latency, and token usage when:
    - All documents are included in context (Full Context)
    - Only relevant documents are retrieved (RAG simulation)
    """
    
    NAME = "RAG Impact"
    DESCRIPTION = "Compares RAG vs Full Context retrieval strategies"
    ESTIMATED_DURATION = "~25 minutes"
    DIFFICULTY = "Medium+"
    
    def __init__(
        self,
        config=None,
        verbose: bool = True,
        total_documents: int = 20,
        relevant_documents: int = 3,
        words_per_doc: int = 200,
    ):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object
            verbose: Whether to print progress
            total_documents: Total documents in the corpus
            relevant_documents: Number of relevant documents for RAG
            words_per_doc: Words per document
        """
        super().__init__(config, verbose)
        self.total_documents = total_documents
        self.relevant_documents = relevant_documents
        self.words_per_doc = words_per_doc
        self.doc_generator = DocumentGenerator(seed=self.config.random_seed)
        self.token_counter = TokenCounter()
        
        # Storage for results
        self.full_context_results: list[dict] = []
        self.rag_results: list[dict] = []
    
    def setup(self) -> None:
        """Set up the experiment."""
        self.log(f"Total documents: {self.total_documents}")
        self.log(f"RAG retrieves: {self.relevant_documents} documents")
        
        if not self.llm_client.is_available():
            raise RuntimeError("LLM client not available. Check API key.")
    
    def _create_document_corpus(self) -> tuple[list[str], str, str, int]:
        """
        Create a document corpus with one relevant document.
        
        Returns:
            Tuple of (all_documents, query, expected_answer, relevant_doc_index)
        """
        documents = []
        relevant_index = self.total_documents // 2  # Put relevant doc in middle
        
        fact_text = "The company's annual revenue reached 50 million dollars."
        expected = "50 million dollars"
        
        for i in range(self.total_documents):
            if i == relevant_index:
                # This is the relevant document
                filler = self.doc_generator.generate_filler_text(
                    self.words_per_doc - len(fact_text.split())
                )
                doc = f"{fact_text} {filler}"
            else:
                # Filler document
                doc = self.doc_generator.generate_filler_text(self.words_per_doc)
            
            documents.append(f"Document {i + 1}:\n{doc}")
        
        query = "What is the company's annual revenue? Answer with just the amount."
        
        return documents, query, expected, relevant_index
    
    def _simulate_rag_retrieval(
        self, 
        documents: list[str], 
        relevant_index: int
    ) -> list[str]:
        """
        Simulate RAG by returning the relevant document + some neighbors.
        
        In a real implementation, this would use vector similarity search.
        Here we simulate it by returning documents near the relevant one.
        """
        # Get the relevant document and a few neighbors
        start = max(0, relevant_index - 1)
        end = min(len(documents), relevant_index + self.relevant_documents - 1)
        
        return documents[start:end]
    
    def run_trial(self, trial_num: int, **kwargs) -> dict:
        """
        Run a single trial comparing both approaches.
        
        Returns:
            Dictionary with results for both approaches
        """
        # Generate document corpus
        documents, query, expected, relevant_index = self._create_document_corpus()
        
        # === Full Context Approach ===
        full_context = "\n\n".join(documents)
        full_tokens = self.token_counter.count(full_context)
        
        full_response = self.llm_client.query(full_context, query)
        full_correct, full_confidence = self.llm_client.evaluate_response(
            full_response.content, expected
        )
        
        full_result = {
            "trial": trial_num,
            "correct": full_correct,
            "confidence": full_confidence,
            "latency": full_response.latency,
            "input_tokens": full_response.input_tokens,
            "context_tokens": full_tokens,
        }
        self.full_context_results.append(full_result)
        
        # === RAG Approach ===
        rag_docs = self._simulate_rag_retrieval(documents, relevant_index)
        rag_context = "\n\n".join(rag_docs)
        rag_tokens = self.token_counter.count(rag_context)
        
        rag_response = self.llm_client.query(rag_context, query)
        rag_correct, rag_confidence = self.llm_client.evaluate_response(
            rag_response.content, expected
        )
        
        rag_result = {
            "trial": trial_num,
            "correct": rag_correct,
            "confidence": rag_confidence,
            "latency": rag_response.latency,
            "input_tokens": rag_response.input_tokens,
            "context_tokens": rag_tokens,
        }
        self.rag_results.append(rag_result)
        
        return {
            "full_context": full_result,
            "rag": rag_result,
        }
    
    def analyze(self) -> dict:
        """
        Analyze results comparing both approaches.
        
        Returns:
            Statistical comparison of RAG vs Full Context
        """
        # Calculate metrics for each approach
        full_accuracies = [1.0 if r["correct"] else 0.0 for r in self.full_context_results]
        rag_accuracies = [1.0 if r["correct"] else 0.0 for r in self.rag_results]
        
        full_latencies = [r["latency"] for r in self.full_context_results]
        rag_latencies = [r["latency"] for r in self.rag_results]
        
        full_tokens = [r["context_tokens"] for r in self.full_context_results]
        rag_tokens = [r["context_tokens"] for r in self.rag_results]
        
        # Aggregate metrics
        metrics = {
            "full": {
                "accuracy": np.mean(full_accuracies),
                "latency": np.mean(full_latencies),
                "tokens": np.mean(full_tokens),
            },
            "rag": {
                "accuracy": np.mean(rag_accuracies),
                "latency": np.mean(rag_latencies),
                "tokens": np.mean(rag_tokens),
            },
        }
        
        # Statistical comparison
        accuracy_comparison = self.analyzer.independent_t_test(
            full_accuracies, rag_accuracies,
            "Full Context", "RAG"
        )
        
        latency_comparison = self.analyzer.independent_t_test(
            full_latencies, rag_latencies,
            "Full Context", "RAG"
        )
        
        # Calculate improvements
        accuracy_improvement = metrics["rag"]["accuracy"] - metrics["full"]["accuracy"]
        latency_improvement = metrics["full"]["latency"] - metrics["rag"]["latency"]
        token_savings = metrics["full"]["tokens"] - metrics["rag"]["tokens"]
        
        main_finding = (
            f"RAG improves accuracy by {accuracy_improvement:.1%} "
            f"({metrics['rag']['accuracy']:.1%} vs {metrics['full']['accuracy']:.1%}). "
            f"Latency reduced by {latency_improvement:.2f}s. "
            f"Token usage reduced by {token_savings:.0f} tokens ({token_savings/metrics['full']['tokens']*100:.0f}% savings)."
        )
        
        return {
            "metrics": metrics,
            "comparisons": {
                "accuracy": {
                    "test": accuracy_comparison.test_name,
                    "statistic": accuracy_comparison.statistic,
                    "p_value": accuracy_comparison.p_value,
                    "significant": accuracy_comparison.is_significant,
                },
                "latency": {
                    "test": latency_comparison.test_name,
                    "statistic": latency_comparison.statistic,
                    "p_value": latency_comparison.p_value,
                    "significant": latency_comparison.is_significant,
                },
            },
            "improvements": {
                "accuracy": accuracy_improvement,
                "latency": latency_improvement,
                "tokens": token_savings,
            },
            "main_finding": main_finding,
            "key_metrics": {
                "Full Context Accuracy": f"{metrics['full']['accuracy']:.1%}",
                "RAG Accuracy": f"{metrics['rag']['accuracy']:.1%}",
                "Latency Improvement": f"{latency_improvement:.2f}s",
                "Token Savings": f"{token_savings:.0f} ({token_savings/metrics['full']['tokens']*100:.0f}%)",
            },
            "statistical_significance": accuracy_comparison.interpretation,
        }
    
    def visualize(self, output_dir: Path) -> list[Path]:
        """Generate visualizations of results."""
        output_dir = Path(output_dir)
        
        # Aggregate metrics
        full_accuracies = [1.0 if r["correct"] else 0.0 for r in self.full_context_results]
        rag_accuracies = [1.0 if r["correct"] else 0.0 for r in self.rag_results]
        
        metrics = {
            "full": {
                "accuracy": np.mean(full_accuracies),
                "latency": np.mean([r["latency"] for r in self.full_context_results]),
                "tokens": np.mean([r["context_tokens"] for r in self.full_context_results]),
            },
            "rag": {
                "accuracy": np.mean(rag_accuracies),
                "latency": np.mean([r["latency"] for r in self.rag_results]),
                "tokens": np.mean([r["context_tokens"] for r in self.rag_results]),
            },
        }
        
        # Create the plot
        plot_path = output_dir / "rag_comparison.png"
        self.visualizer.plot_rag_comparison(
            metrics,
            title=f"RAG vs Full Context ({self.total_documents} Documents)",
            save_path=plot_path,
        )
        
        return [plot_path]
