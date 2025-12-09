"""
Base experiment class for Context Windows Lab.

Provides common functionality for all experiments.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.visualization import Visualizer
from ..utils.statistics import StatisticalAnalyzer
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


@dataclass
class ExperimentResult:
    """
    Container for experiment results.
    
    Attributes:
        experiment_name: Name of the experiment
        timestamp: When the experiment was run
        config: Configuration used
        raw_results: Raw experimental data
        analysis: Statistical analysis results
        visualizations: Paths to generated plots
        summary: Human-readable summary
    """
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict = field(default_factory=dict)
    raw_results: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    visualizations: list[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return asdict(self)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            # הוספנו את הפרמטר cls=NumpyEncoder
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class BaseExperiment(ABC):
    """
    Abstract base class for experiments.
    
    Provides common setup, execution flow, and result handling.
    Subclasses must implement the `run_trial` and `analyze` methods.
    """
    
    # Experiment metadata (override in subclasses)
    NAME: str = "Base Experiment"
    DESCRIPTION: str = "Base experiment class"
    ESTIMATED_DURATION: str = "Unknown"
    DIFFICULTY: str = "Unknown"
    
    def __init__(
        self,
        config: Optional[Config] = None,
        verbose: bool = True,
    ):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object (uses default if None)
            verbose: Whether to print progress messages
        """
        self.config = config or Config.from_env()
        self.verbose = verbose
        
        # Initialize utilities
        self.llm_client = LLMClient(self.config)
        self.visualizer = Visualizer()
        self.analyzer = StatisticalAnalyzer()
        
        # Results storage
        self.results: dict[str, Any] = {}
    
    def log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.NAME}] {message}")
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the experiment.
        
        Override to initialize experiment-specific resources.
        """
        pass
    
    @abstractmethod
    def run_trial(self, trial_num: int, **kwargs) -> dict:
        """
        Run a single trial of the experiment.
        
        Args:
            trial_num: The trial number (1-indexed)
            **kwargs: Additional trial parameters
            
        Returns:
            Dictionary with trial results
        """
        pass
    
    @abstractmethod
    def analyze(self) -> dict:
        """
        Analyze the collected results.
        
        Returns:
            Dictionary with analysis results
        """
        pass
    
    @abstractmethod
    def visualize(self, output_dir: Path) -> list[Path]:
        """
        Generate visualizations of the results.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plots
        """
        pass
    
    def run(
        self,
        num_trials: Optional[int] = None,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> ExperimentResult:
        """
        Run the complete experiment.
        
        Args:
            num_trials: Number of trials (uses config default if None)
            output_dir: Directory to save results (uses config default if None)
            **kwargs: Additional parameters passed to run_trial
            
        Returns:
            ExperimentResult with all data and analysis
        """
        num_trials = num_trials or self.config.num_runs
        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Starting experiment: {self.NAME}")
        self.log(f"Configuration: {self.config}")
        self.log(f"Number of trials: {num_trials}")
        
        # Setup
        self.log("Setting up experiment...")
        self.setup()
        
        # Run trials
        self.log("Running trials...")
        trial_results = []
        for i in range(1, num_trials + 1):
            self.log(f"  Trial {i}/{num_trials}")
            try:
                result = self.run_trial(i, **kwargs)
                trial_results.append(result)
            except Exception as e:
                self.log(f"  Error in trial {i}: {e}")
                trial_results.append({"error": str(e), "trial": i})
        
        self.results["trials"] = trial_results
        
        # Analyze
        self.log("Analyzing results...")
        analysis = self.analyze()
        
        # Visualize
        self.log("Generating visualizations...")
        viz_paths = self.visualize(output_dir)
        
        # Create result object
        result = ExperimentResult(
            experiment_name=self.NAME,
            config={
                "model": self.config.claude_model,
                "num_trials": num_trials,
                "random_seed": self.config.random_seed,
            },
            raw_results=self.results,
            analysis=analysis,
            visualizations=[str(p) for p in viz_paths],
            summary=self._generate_summary(analysis),
        )
        
        # Save results
        result_path = output_dir / f"{self.NAME.lower().replace(' ', '_')}_results.json"
        result.save(result_path)
        self.log(f"Results saved to: {result_path}")
        
        return result
    
    def _generate_summary(self, analysis: dict) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = [
            f"Experiment: {self.NAME}",
            f"Description: {self.DESCRIPTION}",
            "-" * 50,
        ]
        
        if "main_finding" in analysis:
            lines.append(f"Main Finding: {analysis['main_finding']}")
        
        if "key_metrics" in analysis:
            lines.append("\nKey Metrics:")
            for metric, value in analysis["key_metrics"].items():
                lines.append(f"  - {metric}: {value}")
        
        if "statistical_significance" in analysis:
            lines.append(f"\nStatistical Significance: {analysis['statistical_significance']}")
        
        return "\n".join(lines)
