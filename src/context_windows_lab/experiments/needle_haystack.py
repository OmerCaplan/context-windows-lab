"""
Experiment 1: Needle in Haystack (Lost in the Middle)

Demonstrates that LLMs have difficulty retrieving information 
placed in the middle of a long context, compared to the beginning or end.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseExperiment
from ..utils.document_generator import DocumentGenerator, FactPosition


class NeedleInHaystackExperiment(BaseExperiment):
    """
    Tests the "Lost in the Middle" phenomenon.
    
    Places critical facts at different positions (start/middle/end)
    in a context and measures retrieval accuracy for each position.
    
    Expected Result: High accuracy at start/end, low accuracy in middle.
    """
    
    NAME = "Needle in Haystack"
    DESCRIPTION = "Demonstrates the Lost in the Middle phenomenon"
    ESTIMATED_DURATION = "~15 minutes"
    DIFFICULTY = "Basic"
    
    def __init__(
        self,
        config=None,
        verbose: bool = True,
        num_documents: int = 5,
        words_per_doc: int = 200,
    ):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object
            verbose: Whether to print progress
            num_documents: Number of documents in the haystack
            words_per_doc: Words per document
        """
        super().__init__(config, verbose)
        self.num_documents = num_documents
        self.words_per_doc = words_per_doc
        self.doc_generator = DocumentGenerator(seed=self.config.random_seed)
        
        # Storage for results by position
        self.position_results: dict[str, list[dict]] = {
            "start": [],
            "middle": [],
            "end": [],
        }
    
    def setup(self) -> None:
        """Set up the experiment."""
        self.log(f"Haystack size: {self.num_documents} documents")
        self.log(f"Words per document: {self.words_per_doc}")
        
        if not self.llm_client.is_available():
            raise RuntimeError("LLM client not available. Check API key.")
    
    def run_trial(self, trial_num: int, **kwargs) -> dict:
        """
        Run a single trial testing all three positions.
        
        Returns:
            Dictionary with accuracy for each position
        """
        trial_result = {}
        
        for position in [FactPosition.START, FactPosition.MIDDLE, FactPosition.END]:
            # Generate context with needle at specified position
            context, query, expected = self.doc_generator.create_needle_haystack_context(
                num_documents=self.num_documents,
                needle_position=position,
                words_per_doc=self.words_per_doc,
            )
            
            # Query the LLM
            response = self.llm_client.query(context, query)
            
            # Evaluate
            is_correct, confidence = self.llm_client.evaluate_response(
                response.content, expected
            )
            
            position_name = position.value
            trial_result[position_name] = {
                "correct": is_correct,
                "confidence": confidence,
                "latency": response.latency,
                "tokens": response.total_tokens,
                "response": response.content[:100],  # Truncate for storage
            }
            
            self.position_results[position_name].append({
                "trial": trial_num,
                "correct": is_correct,
                "confidence": confidence,
                "latency": response.latency,
            })
        
        return trial_result
    
    def analyze(self) -> dict:
        """
        Analyze results across all trials.
        
        Returns:
            Statistical analysis of accuracy by position
        """
        # Calculate accuracy for each position
        accuracies = {}
        for position in ["start", "middle", "end"]:
            results = self.position_results[position]
            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            accuracies[position] = correct / total if total > 0 else 0
        
        # Get detailed stats
        position_data = {
            pos: [1.0 if r["correct"] else 0.0 for r in self.position_results[pos]]
            for pos in ["start", "middle", "end"]
        }
        
        # Statistical comparison
        stats_summary = self.analyzer.summarize_experiment(position_data, "accuracy")
        
        # Main finding
        middle_accuracy = accuracies["middle"]
        edge_accuracy = (accuracies["start"] + accuracies["end"]) / 2
        
        main_finding = (
            f"The 'Lost in the Middle' effect is "
            f"{'confirmed' if middle_accuracy < edge_accuracy * 0.8 else 'not clearly observed'}. "
            f"Middle accuracy ({middle_accuracy:.1%}) is "
            f"{'significantly lower' if middle_accuracy < edge_accuracy * 0.8 else 'similar'} "
            f"compared to edge positions ({edge_accuracy:.1%})."
        )
        
        return {
            "accuracies": accuracies,
            "detailed_stats": stats_summary,
            "main_finding": main_finding,
            "key_metrics": {
                "Start Accuracy": f"{accuracies['start']:.1%}",
                "Middle Accuracy": f"{accuracies['middle']:.1%}",
                "End Accuracy": f"{accuracies['end']:.1%}",
                "Edge vs Middle Difference": f"{(edge_accuracy - middle_accuracy):.1%}",
            },
            "statistical_significance": (
                stats_summary.get("statistical_tests", [{}])[0].get("interpretation", "N/A")
                if stats_summary.get("statistical_tests") else "N/A"
            ),
        }
    
    def visualize(self, output_dir: Path) -> list[Path]:
        """Generate visualization of results."""
        output_dir = Path(output_dir)
        
        # Prepare data for plotting
        results_for_plot = {
            pos: [1.0 if r["correct"] else 0.0 for r in self.position_results[pos]]
            for pos in ["start", "middle", "end"]
        }
        
        # Create the plot
        plot_path = output_dir / "needle_haystack_accuracy.png"
        self.visualizer.plot_accuracy_by_position(
            results_for_plot,
            title=f"Accuracy by Fact Position ({self.num_documents} Documents)",
            save_path=plot_path,
        )
        
        return [plot_path]
