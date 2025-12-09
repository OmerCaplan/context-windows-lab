"""
Experiment 2: Context Window Size Impact

Demonstrates how increasing context size affects:
- Response accuracy
- Latency
- Token usage

Expected: Accuracy decreases and latency increases as context grows.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseExperiment
from ..utils.document_generator import DocumentGenerator, FactPosition
from ..utils.token_counter import TokenCounter


class ContextSizeExperiment(BaseExperiment):
    """
    Tests the impact of context window size on LLM performance.
    
    Gradually increases the number of documents in the context
    and measures accuracy, latency, and token usage at each level.
    """
    
    NAME = "Context Size Impact"
    DESCRIPTION = "Shows how context window size affects accuracy and latency"
    ESTIMATED_DURATION = "~20 minutes"
    DIFFICULTY = "Medium"
    
    def __init__(
        self,
        config=None,
        verbose: bool = True,
        doc_counts: Optional[list[int]] = None,
        words_per_doc: int = 200,
    ):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object
            verbose: Whether to print progress
            doc_counts: List of document counts to test
            words_per_doc: Words per document
        """
        super().__init__(config, verbose)
        self.doc_counts = doc_counts or [2, 5, 10, 20, 50]
        self.words_per_doc = words_per_doc
        self.doc_generator = DocumentGenerator(seed=self.config.random_seed)
        self.token_counter = TokenCounter()
        
        # Storage for results by document count
        self.size_results: dict[int, list[dict]] = {n: [] for n in self.doc_counts}
    
    def setup(self) -> None:
        """Set up the experiment."""
        self.log(f"Document counts to test: {self.doc_counts}")
        self.log(f"Words per document: {self.words_per_doc}")
        
        if not self.llm_client.is_available():
            raise RuntimeError("LLM client not available. Check API key.")
    
    def run_trial(self, trial_num: int, **kwargs) -> dict:
        """
        Run a single trial testing all document counts.
        
        Returns:
            Dictionary with results for each document count
        """
        trial_result = {}
        
        for num_docs in self.doc_counts:
            # Generate context with needle in middle
            context, query, expected = self.doc_generator.create_needle_haystack_context(
                num_documents=num_docs,
                needle_position=FactPosition.MIDDLE,
                words_per_doc=self.words_per_doc,
            )
            
            # Count tokens
            token_count = self.token_counter.count(context)
            
            # Query the LLM
            response = self.llm_client.query(context, query)
            
            # Evaluate
            is_correct, confidence = self.llm_client.evaluate_response(
                response.content, expected
            )
            
            result = {
                "correct": is_correct,
                "confidence": confidence,
                "latency": response.latency,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "context_tokens": token_count,
            }
            
            trial_result[num_docs] = result
            self.size_results[num_docs].append({
                "trial": trial_num,
                **result,
            })
        
        return trial_result
    
    def analyze(self) -> dict:
        """
        Analyze results across all trials and document counts.
        
        Returns:
            Statistical analysis of performance vs context size
        """
        # Aggregate metrics by document count
        aggregated = {}
        for num_docs in self.doc_counts:
            results = self.size_results[num_docs]
            if not results:
                continue
            
            accuracies = [1.0 if r["correct"] else 0.0 for r in results]
            latencies = [r["latency"] for r in results]
            tokens = [r["context_tokens"] for r in results]
            
            aggregated[num_docs] = {
                "accuracy": {
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies),
                },
                "latency": {
                    "mean": np.mean(latencies),
                    "std": np.std(latencies),
                },
                "tokens": {
                    "mean": np.mean(tokens),
                    "std": np.std(tokens),
                },
            }
        
        # Calculate correlation between context size and accuracy
        doc_count_list = list(aggregated.keys())
        accuracy_list = [aggregated[n]["accuracy"]["mean"] for n in doc_count_list]
        latency_list = [aggregated[n]["latency"]["mean"] for n in doc_count_list]
        
        accuracy_correlation = self.analyzer.correlation_analysis(
            doc_count_list, accuracy_list,
            "Context Size", "Accuracy"
        )
        
        latency_correlation = self.analyzer.correlation_analysis(
            doc_count_list, latency_list,
            "Context Size", "Latency"
        )
        
        # Determine main finding
        accuracy_decline = accuracy_list[0] - accuracy_list[-1] if accuracy_list else 0
        
        main_finding = (
            f"Accuracy decreases by {accuracy_decline:.1%} as context grows from "
            f"{doc_count_list[0]} to {doc_count_list[-1]} documents. "
            f"Latency increases from {latency_list[0]:.2f}s to {latency_list[-1]:.2f}s. "
            f"Accuracy-size correlation: r={accuracy_correlation.statistic:.3f} "
            f"({'significant' if accuracy_correlation.is_significant else 'not significant'})."
        )
        
        return {
            "aggregated": aggregated,
            "accuracy_correlation": {
                "r": accuracy_correlation.statistic,
                "p_value": accuracy_correlation.p_value,
                "significant": accuracy_correlation.is_significant,
            },
            "latency_correlation": {
                "r": latency_correlation.statistic,
                "p_value": latency_correlation.p_value,
                "significant": latency_correlation.is_significant,
            },
            "main_finding": main_finding,
            "key_metrics": {
                "Min Context Accuracy": f"{accuracy_list[0]:.1%}" if accuracy_list else "N/A",
                "Max Context Accuracy": f"{accuracy_list[-1]:.1%}" if accuracy_list else "N/A",
                "Accuracy Decline": f"{accuracy_decline:.1%}",
                "Latency Increase": f"{(latency_list[-1] - latency_list[0]):.2f}s" if latency_list else "N/A",
            },
            "statistical_significance": accuracy_correlation.interpretation,
        }
    
    def visualize(self, output_dir: Path) -> list[Path]:
        """Generate visualizations of results."""
        output_dir = Path(output_dir)
        
        # Aggregate data for plotting
        doc_counts = []
        accuracies = []
        latencies = []
        tokens = []
        
        for num_docs in self.size_results.keys():
            results = self.size_results[num_docs]
            # Skip if no valid results (all errors)
            valid_results = [r for r in results if "error" not in r]
            if not valid_results:
                continue
            
            doc_counts.append(num_docs)
            accuracies.append(np.mean([1.0 if r.get("correct", False) else 0.0 for r in valid_results]))
            latencies.append(np.mean([r.get("latency", 0) for r in valid_results]))
            tokens.append(np.mean([r.get("context_tokens", 0) for r in valid_results]))
        
        # Handle case where no valid results exist
        if not doc_counts:
            self.log("No valid results to visualize")
            plot_path = output_dir / "context_size_impact.png"
            # Create empty placeholder plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No valid results\n(API errors occurred)", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            fig.savefig(plot_path)
            plt.close(fig)
            return [plot_path]
        
        # Handle NaN values
        tokens = [int(t) if not np.isnan(t) else 0 for t in tokens]
        accuracies = [a if not np.isnan(a) else 0 for a in accuracies]
        latencies = [l if not np.isnan(l) else 0 for l in latencies]
        
        # Create the plot
        plot_path = output_dir / "context_size_impact.png"
        self.visualizer.plot_context_size_impact(
            doc_counts=doc_counts,
            accuracies=accuracies,
            latencies=latencies,
            tokens=tokens,
            title="Context Window Size Impact on Performance",
            save_path=plot_path,
        )
        
        return [plot_path]
