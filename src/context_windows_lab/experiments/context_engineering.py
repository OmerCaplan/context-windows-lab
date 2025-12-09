"""
Experiment 4: Context Engineering Strategies

Tests different strategies for managing context in multi-step agent workflows:
1. SELECT: Use RAG to retrieve only relevant history
2. COMPRESS: Summarize history when it gets too long
3. WRITE: Use external scratchpad for key facts
4. BASELINE: No strategy (accumulate everything)

Expected: Strategies maintain performance as history grows.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseExperiment
from ..utils.document_generator import DocumentGenerator
from ..utils.token_counter import TokenCounter


class ContextEngineeringExperiment(BaseExperiment):
    """
    Tests context engineering strategies for multi-step agents.
    
    Simulates an agent executing multiple actions, with history
    accumulating after each step. Compares different strategies
    for managing this growing context.
    """
    
    NAME = "Context Engineering"
    DESCRIPTION = "Tests context management strategies (Select, Compress, Write)"
    ESTIMATED_DURATION = "~30 minutes"
    DIFFICULTY = "Advanced"
    
    STRATEGIES = ["baseline", "select", "compress", "write"]
    
    def __init__(
        self,
        config=None,
        verbose: bool = True,
        num_actions: int = 10,
        action_output_words: int = 100,
        max_context_tokens: int = 2000,
    ):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object
            verbose: Whether to print progress
            num_actions: Number of sequential actions to simulate
            action_output_words: Words generated per action
            max_context_tokens: Token limit for compress strategy
        """
        super().__init__(config, verbose)
        self.num_actions = num_actions
        self.action_output_words = action_output_words
        self.max_context_tokens = max_context_tokens
        self.doc_generator = DocumentGenerator(seed=self.config.random_seed)
        self.token_counter = TokenCounter()
        
        # Storage for results by strategy
        self.strategy_results: dict[str, list[dict]] = {
            s: [] for s in self.STRATEGIES
        }
    
    def setup(self) -> None:
        """Set up the experiment."""
        self.log(f"Number of actions: {self.num_actions}")
        self.log(f"Strategies to test: {self.STRATEGIES}")
        
        if not self.llm_client.is_available():
            raise RuntimeError("LLM client not available. Check API key.")
    
    def _generate_action_output(self, action_num: int) -> str:
        """Generate simulated output from an agent action."""
        outputs = [
            f"Action {action_num} completed successfully.",
            f"Retrieved data for step {action_num}.",
            f"Processed item {action_num} with status: complete.",
            f"Updated record {action_num} in the database.",
            f"Sent notification for action {action_num}.",
        ]
        
        base = outputs[action_num % len(outputs)]
        filler = self.doc_generator.generate_filler_text(self.action_output_words - 10)
        return f"{base} {filler}"
    
    def _apply_baseline_strategy(self, history: list[str]) -> str:
        """Baseline: just concatenate all history."""
        return "\n\n".join(history)
    
    def _apply_select_strategy(self, history: list[str], k: int = 3) -> str:
        """Select: only keep the most recent k items."""
        selected = history[-k:] if len(history) > k else history
        return "\n\n".join(selected)
    
    def _apply_compress_strategy(self, history: list[str]) -> str:
        """Compress: summarize if history is too long."""
        full_history = "\n\n".join(history)
        tokens = self.token_counter.count(full_history)
        
        if tokens > self.max_context_tokens:
            # Simulate compression by taking first and last items + summary note
            compressed = (
                f"[Summary of {len(history) - 2} earlier actions: "
                f"Actions completed successfully with various data operations.]\n\n"
                f"{history[0]}\n\n...\n\n{history[-1]}"
            )
            return compressed
        
        return full_history
    
    def _apply_write_strategy(self, history: list[str]) -> str:
        """Write: extract key facts into a scratchpad."""
        # Simulate scratchpad with key facts
        scratchpad = "SCRATCHPAD:\n"
        for i, item in enumerate(history):
            # Extract a "key fact" (first sentence)
            first_sentence = item.split(".")[0] + "."
            scratchpad += f"- {first_sentence}\n"
        
        scratchpad += "\nMOST RECENT ACTION:\n" + history[-1]
        return scratchpad
    
    def _create_test_query(self, action_num: int) -> tuple[str, str]:
        """Create a query about the action history."""
        query = f"How many actions have been completed so far? Answer with just the number."
        expected = str(action_num)
        return query, expected
    
    def run_trial(self, trial_num: int, **kwargs) -> dict:
        """
        Run a single trial testing all strategies.
        
        Returns:
            Dictionary with results for each strategy
        """
        trial_result = {}
        
        for strategy in self.STRATEGIES:
            history = []
            strategy_metrics = {
                "accuracies": [],
                "latencies": [],
                "tokens": [],
            }
            
            for action_num in range(1, self.num_actions + 1):
                # Generate action output
                output = self._generate_action_output(action_num)
                history.append(f"Action {action_num} Output:\n{output}")
                
                # Apply strategy
                if strategy == "baseline":
                    context = self._apply_baseline_strategy(history)
                elif strategy == "select":
                    context = self._apply_select_strategy(history)
                elif strategy == "compress":
                    context = self._apply_compress_strategy(history)
                else:  # write
                    context = self._apply_write_strategy(history)
                
                # Create query
                query, expected = self._create_test_query(action_num)
                
                # Query LLM
                response = self.llm_client.query(context, query)
                is_correct, _ = self.llm_client.evaluate_response(
                    response.content, expected
                )
                
                strategy_metrics["accuracies"].append(1.0 if is_correct else 0.0)
                strategy_metrics["latencies"].append(response.latency)
                strategy_metrics["tokens"].append(response.input_tokens)
            
            # Aggregate metrics for this strategy
            result = {
                "trial": trial_num,
                "accuracy": np.mean(strategy_metrics["accuracies"]),
                "latency": np.mean(strategy_metrics["latencies"]),
                "tokens": np.mean(strategy_metrics["tokens"]),
                "final_accuracy": strategy_metrics["accuracies"][-1],
                "accuracy_trend": strategy_metrics["accuracies"],
            }
            
            trial_result[strategy] = result
            self.strategy_results[strategy].append(result)
        
        return trial_result
    
    def analyze(self) -> dict:
        """
        Analyze results comparing all strategies.
        
        Returns:
            Statistical comparison of strategies
        """
        # Aggregate metrics by strategy
        metrics = {}
        for strategy in self.STRATEGIES:
            results = self.strategy_results[strategy]
            if not results:
                continue
            
            metrics[strategy] = {
                "accuracy": np.mean([r["accuracy"] for r in results]),
                "latency": np.mean([r["latency"] for r in results]),
                "tokens": np.mean([r["tokens"] for r in results]),
                "final_accuracy": np.mean([r["final_accuracy"] for r in results]),
            }
        
        # Compare strategies using ANOVA
        accuracy_data = {
            s: [r["accuracy"] for r in self.strategy_results[s]]
            for s in self.STRATEGIES
        }
        anova_result = self.analyzer.one_way_anova(accuracy_data)
        
        # Find best strategy
        best_strategy = max(metrics.keys(), key=lambda s: metrics[s]["accuracy"])
        worst_strategy = min(metrics.keys(), key=lambda s: metrics[s]["accuracy"])
        
        main_finding = (
            f"Best strategy: {best_strategy.upper()} (accuracy: {metrics[best_strategy]['accuracy']:.1%}). "
            f"Worst: {worst_strategy.upper()} (accuracy: {metrics[worst_strategy]['accuracy']:.1%}). "
            f"ANOVA: F={anova_result.statistic:.2f}, p={anova_result.p_value:.4f} "
            f"({'significant differences' if anova_result.is_significant else 'no significant differences'})."
        )
        
        return {
            "metrics": metrics,
            "anova": {
                "statistic": anova_result.statistic,
                "p_value": anova_result.p_value,
                "effect_size": anova_result.effect_size,
                "significant": anova_result.is_significant,
            },
            "main_finding": main_finding,
            "key_metrics": {
                f"{s.capitalize()} Accuracy": f"{metrics[s]['accuracy']:.1%}"
                for s in self.STRATEGIES
            },
            "best_strategy": best_strategy,
            "statistical_significance": anova_result.interpretation,
        }
    
    def visualize(self, output_dir: Path) -> list[Path]:
        """Generate visualizations of results."""
        output_dir = Path(output_dir)
        
        # Aggregate metrics
        metrics = {}
        for strategy in self.STRATEGIES:
            results = self.strategy_results[strategy]
            if results:
                metrics[strategy] = {
                    "accuracy": np.mean([r["accuracy"] for r in results]),
                    "latency": np.mean([r["latency"] for r in results]),
                    "tokens": np.mean([r["tokens"] for r in results]),
                }
        
        # Create the plot
        plot_path = output_dir / "context_engineering_comparison.png"
        self.visualizer.plot_strategy_comparison(
            strategies=[s.capitalize() for s in self.STRATEGIES],
            metrics={s.capitalize(): metrics[s] for s in self.STRATEGIES if s in metrics},
            title=f"Context Engineering Strategies ({self.num_actions} Actions)",
            save_path=plot_path,
        )
        
        return [plot_path]
