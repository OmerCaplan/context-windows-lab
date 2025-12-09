"""
Visualization utilities for Context Windows Lab.

Provides consistent, publication-quality plots for experiment results.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


class Visualizer:
    """
    Creates visualizations for experiment results.
    
    Provides consistent styling and formatting for:
    - Accuracy by position plots (Lost in Middle)
    - Context size vs accuracy/latency
    - RAG vs Full Context comparisons
    - Strategy performance comparisons
    """
    
    # Color palette for consistent styling
    COLORS = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "tertiary": "#F18F01",
        "success": "#C73E1D",
        "neutral": "#6B7280",
        "start": "#22C55E",
        "middle": "#EF4444",
        "end": "#3B82F6",
    }
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple = (10, 6)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in style else "default")
        
        # Set up seaborn
        sns.set_palette("husl")
    
    def plot_accuracy_by_position(
        self,
        results: dict[str, list[float]],
        title: str = "Accuracy by Fact Position (Lost in the Middle)",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a bar chart showing accuracy by fact position.
        
        Args:
            results: Dict with keys 'start', 'middle', 'end' and accuracy lists
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        positions = ["Start", "Middle", "End"]
        means = [np.mean(results.get(p.lower(), [0])) * 100 for p in positions]
        stds = [np.std(results.get(p.lower(), [0])) * 100 for p in positions]
        colors = [self.COLORS["start"], self.COLORS["middle"], self.COLORS["end"]]
        
        bars = ax.bar(positions, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_xlabel("Fact Position in Context", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_context_size_impact(
        self,
        doc_counts: list[int],
        accuracies: list[float],
        latencies: list[float],
        tokens: list[int],
        title: str = "Context Window Size Impact",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a dual-axis plot showing accuracy and latency vs context size.
        
        Args:
            doc_counts: Number of documents at each point
            accuracies: Accuracy at each point
            latencies: Latency at each point (seconds)
            tokens: Token count at each point
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Accuracy vs Context Size
        color1 = self.COLORS["primary"]
        ax1.plot(doc_counts, [a * 100 for a in accuracies], "o-", color=color1, linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Documents", fontsize=12)
        ax1.set_ylabel("Accuracy (%)", fontsize=12, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_title("Accuracy Degradation", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Latency vs Context Size  
        color2 = self.COLORS["secondary"]
        ax2.plot(doc_counts, latencies, "s-", color=color2, linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Documents", fontsize=12)
        ax2.set_ylabel("Latency (seconds)", fontsize=12, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_title("Response Time Increase", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        
        # Add token count as secondary x-axis labels
        ax1_2 = ax1.twiny()
        ax1_2.set_xlim(ax1.get_xlim())
        ax1_2.set_xticks(doc_counts)
        ax1_2.set_xticklabels([f"{t//1000}k" for t in tokens], fontsize=9)
        ax1_2.set_xlabel("Approximate Tokens", fontsize=10)
        
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_rag_comparison(
        self,
        metrics: dict[str, dict[str, float]],
        title: str = "RAG vs Full Context Comparison",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a comparison plot for RAG vs Full Context performance.
        
        Args:
            metrics: Dict with 'rag' and 'full' keys, each containing
                    'accuracy', 'latency', 'tokens' metrics
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        methods = ["Full Context", "RAG"]
        colors = [self.COLORS["secondary"], self.COLORS["primary"]]
        
        # Accuracy comparison
        accuracies = [metrics["full"]["accuracy"] * 100, metrics["rag"]["accuracy"] * 100]
        axes[0].bar(methods, accuracies, color=colors, edgecolor="black")
        axes[0].set_ylabel("Accuracy (%)", fontsize=12)
        axes[0].set_title("Accuracy", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 105)
        for i, v in enumerate(accuracies):
            axes[0].annotate(f"{v:.1f}%", xy=(i, v), ha="center", va="bottom", fontweight="bold")
        
        # Latency comparison
        latencies = [metrics["full"]["latency"], metrics["rag"]["latency"]]
        axes[1].bar(methods, latencies, color=colors, edgecolor="black")
        axes[1].set_ylabel("Latency (seconds)", fontsize=12)
        axes[1].set_title("Response Time", fontsize=13, fontweight="bold")
        for i, v in enumerate(latencies):
            axes[1].annotate(f"{v:.2f}s", xy=(i, v), ha="center", va="bottom", fontweight="bold")
        
        # Token usage comparison
        tokens = [metrics["full"]["tokens"], metrics["rag"]["tokens"]]
        axes[2].bar(methods, tokens, color=colors, edgecolor="black")
        axes[2].set_ylabel("Tokens Used", fontsize=12)
        axes[2].set_title("Token Usage", fontsize=13, fontweight="bold")
        for i, v in enumerate(tokens):
            axes[2].annotate(f"{v:,}", xy=(i, v), ha="center", va="bottom", fontweight="bold")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_strategy_comparison(
        self,
        strategies: list[str],
        metrics: dict[str, dict[str, float]],
        title: str = "Context Engineering Strategies Comparison",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a multi-metric comparison of context engineering strategies.
        
        Args:
            strategies: List of strategy names
            metrics: Dict mapping strategy names to their metrics
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = [self.COLORS["primary"], self.COLORS["secondary"], 
                  self.COLORS["tertiary"], self.COLORS["success"]]
        
        # Accuracy
        accuracies = [metrics[s]["accuracy"] * 100 for s in strategies]
        axes[0, 0].bar(strategies, accuracies, color=colors[:len(strategies)], edgecolor="black")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_title("Accuracy by Strategy", fontweight="bold")
        axes[0, 0].set_ylim(0, 105)
        
        # Latency
        latencies = [metrics[s]["latency"] for s in strategies]
        axes[0, 1].bar(strategies, latencies, color=colors[:len(strategies)], edgecolor="black")
        axes[0, 1].set_ylabel("Latency (seconds)")
        axes[0, 1].set_title("Response Time by Strategy", fontweight="bold")
        
        # Token usage
        tokens = [metrics[s]["tokens"] for s in strategies]
        axes[1, 0].bar(strategies, tokens, color=colors[:len(strategies)], edgecolor="black")
        axes[1, 0].set_ylabel("Tokens Used")
        axes[1, 0].set_title("Token Usage by Strategy", fontweight="bold")
        
        # Efficiency score (accuracy / latency)
        efficiency = [a / max(l, 0.01) for a, l in zip(accuracies, latencies)]
        axes[1, 1].bar(strategies, efficiency, color=colors[:len(strategies)], edgecolor="black")
        axes[1, 1].set_ylabel("Efficiency Score")
        axes[1, 1].set_title("Efficiency (Accuracy/Latency)", fontweight="bold")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_experiment_summary(
        self,
        all_results: dict,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a summary dashboard of all experiments.
        
        Args:
            all_results: Dictionary containing results from all experiments
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # This would create a comprehensive summary - simplified version
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Experiment Summary Dashboard\n\nSee individual plots for detailed results",
            ha="center", va="center", fontsize=16, transform=ax.transAxes
        )
        ax.axis("off")
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
