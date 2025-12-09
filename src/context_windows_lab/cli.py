"""
CLI entry point for Context Windows Lab.

Provides command-line interface to run experiments.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import Config
from .experiments import (
    NeedleInHaystackExperiment,
    ContextSizeExperiment,
    RAGImpactExperiment,
    ContextEngineeringExperiment,
)


console = Console()

EXPERIMENTS = {
    "needle": NeedleInHaystackExperiment,
    "context-size": ContextSizeExperiment,
    "rag": RAGImpactExperiment,
    "engineering": ContextEngineeringExperiment,
    "all": None,  # Special case
}


def print_header():
    """Print the application header."""
    console.print(Panel.fit(
        "[bold blue]Context Windows Lab[/bold blue]\n"
        "[dim]LLM Context Window Experiments[/dim]\n"
        "[dim]Team: OmerAndYogever[/dim]",
        border_style="blue",
    ))


def print_experiment_table():
    """Print available experiments."""
    table = Table(title="Available Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Difficulty", style="magenta")
    
    experiments_info = [
        ("needle", "Needle in Haystack", "~15 min", "Basic"),
        ("context-size", "Context Size Impact", "~20 min", "Medium"),
        ("rag", "RAG Impact", "~25 min", "Medium+"),
        ("engineering", "Context Engineering", "~30 min", "Advanced"),
        ("all", "Run All Experiments", "~90 min", "All"),
    ]
    
    for exp_id, name, duration, difficulty in experiments_info:
        table.add_row(exp_id, name, duration, difficulty)
    
    console.print(table)


def run_experiment(exp_id: str, config: Config, num_trials: int, output_dir: Path):
    """Run a single experiment."""
    if exp_id not in EXPERIMENTS:
        console.print(f"[red]Unknown experiment: {exp_id}[/red]")
        return None
    
    exp_class = EXPERIMENTS[exp_id]
    if exp_class is None:
        return None
    
    console.print(f"\n[bold]Running: {exp_class.NAME}[/bold]")
    console.print(f"[dim]{exp_class.DESCRIPTION}[/dim]\n")
    
    experiment = exp_class(config=config, verbose=True)
    result = experiment.run(num_trials=num_trials, output_dir=output_dir)
    
    console.print(f"\n[green]✓ {exp_class.NAME} completed![/green]")
    console.print(f"[dim]Results saved to: {output_dir}[/dim]")
    
    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Context Windows Lab - LLM Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  context-lab needle              Run Needle in Haystack experiment
  context-lab all --trials 3      Run all experiments with 3 trials each
  context-lab rag --output ./out  Run RAG experiment, save to ./out
        """,
    )
    
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run (needle, context-size, rag, engineering, all)",
    )
    
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=5,
        help="Number of trials per experiment (default: 5)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for results (default: ./outputs)",
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiments",
    )
    
    args = parser.parse_args()
    
    print_header()
    
    if args.list or not args.experiment:
        print_experiment_table()
        return 0
    
    # Load configuration
    config = Config.from_env()
    
    # Validate
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  [red]• {error}[/red]")
        return 1
    
    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run experiment(s)
    if args.experiment == "all":
        results = []
        for exp_id in ["needle", "context-size", "rag", "engineering"]:
            result = run_experiment(exp_id, config, args.trials, args.output)
            if result:
                results.append(result)
        
        console.print(f"\n[bold green]All experiments completed![/bold green]")
        console.print(f"[dim]Results saved to: {args.output}[/dim]")
    else:
        run_experiment(args.experiment, config, args.trials, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
