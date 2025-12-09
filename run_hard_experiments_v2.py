#!/usr/bin/env python
"""
Run harder experiments to demonstrate context window phenomena.

This version includes:
- Rate limiting delays to avoid API errors
- Even harder parameters (Haiku is very capable!)
- Better error handling
"""

import sys
import time
sys.path.insert(0, 'src')

from pathlib import Path
from context_windows_lab.config import Config
from context_windows_lab.experiments import (
    NeedleInHaystackExperiment,
    ContextSizeExperiment,
    RAGImpactExperiment,
    ContextEngineeringExperiment,
)


# Delay between API calls (seconds) to avoid rate limiting
API_DELAY = 3


def patch_experiment_with_delay(experiment_class):
    """Add delay to experiment's run_trial to avoid rate limits."""
    original_run_trial = experiment_class.run_trial
    
    def delayed_run_trial(self, trial_num, **kwargs):
        if trial_num > 1:
            print(f"[{self.NAME}]   Waiting {API_DELAY}s to avoid rate limits...")
            time.sleep(API_DELAY)
        return original_run_trial(self, trial_num, **kwargs)
    
    experiment_class.run_trial = delayed_run_trial
    return experiment_class


def main():
    # Load config
    config = Config.from_env()
    
    # Create separate output directory for hard experiments
    output_dir = Path('./outputs_hard')
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("RUNNING HARDER EXPERIMENTS (with rate limiting)")
    print("="*70)
    print(f"Model: {config.claude_model}")
    print(f"Output directory: {output_dir}")
    print(f"Delay between trials: {API_DELAY}s")
    print("="*70)
    
    # Patch experiments with delay
    NeedleInHaystackExperiment_Delayed = patch_experiment_with_delay(NeedleInHaystackExperiment)
    ContextSizeExperiment_Delayed = patch_experiment_with_delay(ContextSizeExperiment)
    RAGImpactExperiment_Delayed = patch_experiment_with_delay(RAGImpactExperiment)
    ContextEngineeringExperiment_Delayed = patch_experiment_with_delay(ContextEngineeringExperiment)
    
    # =========================================================================
    # Experiment 1: Needle in Haystack (EVEN HARDER)
    # - 30 documents (Haiku got 100% with 15!)
    # - 500 words per document
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Needle in Haystack (HARD MODE)")
    print("  - 30 documents (increased from 15)")
    print("  - 500 words per document")
    print("="*70)
    
    try:
        needle_exp = NeedleInHaystackExperiment_Delayed(
            config=config,
            verbose=True,
            num_documents=30,      # Even more documents
            words_per_doc=500,     # Longer documents
        )
        needle_result = needle_exp.run(num_trials=3, output_dir=output_dir)  # Fewer trials
        print(f"\n>>> RESULT: {needle_result.analysis.get('main_finding', 'N/A')}")
    except Exception as e:
        print(f"\n>>> ERROR in Experiment 1: {e}")
        needle_result = None
    
    # Wait before next experiment
    print("\nWaiting 30s before next experiment...")
    time.sleep(30)
    
    # =========================================================================
    # Experiment 2: Context Size Impact (REDUCED to avoid rate limits)
    # - Fewer, larger jumps: [10, 30, 60]
    # - 500 words per document
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Context Size Impact (HARD MODE)")
    print("  - Document counts: [10, 30, 60] (reduced to avoid rate limits)")
    print("  - 500 words per document")
    print("="*70)
    
    try:
        size_exp = ContextSizeExperiment_Delayed(
            config=config,
            verbose=True,
            doc_counts=[10, 30, 60],   # Fewer steps to avoid rate limits
            words_per_doc=500,
        )
        size_result = size_exp.run(num_trials=3, output_dir=output_dir)
        print(f"\n>>> RESULT: {size_result.analysis.get('main_finding', 'N/A')}")
    except Exception as e:
        print(f"\n>>> ERROR in Experiment 2: {e}")
        size_result = None
    
    # Wait before next experiment
    print("\nWaiting 30s before next experiment...")
    time.sleep(30)
    
    # =========================================================================
    # Experiment 3: RAG Impact (HARDER)
    # - 40 documents
    # - 500 words per document
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: RAG Impact (HARD MODE)")
    print("  - 40 total documents")
    print("  - 3 relevant documents for RAG")
    print("  - 500 words per document")
    print("="*70)
    
    try:
        rag_exp = RAGImpactExperiment_Delayed(
            config=config,
            verbose=True,
            total_documents=40,
            relevant_documents=3,
            words_per_doc=500,
        )
        rag_result = rag_exp.run(num_trials=3, output_dir=output_dir)
        print(f"\n>>> RESULT: {rag_result.analysis.get('main_finding', 'N/A')}")
    except Exception as e:
        print(f"\n>>> ERROR in Experiment 3: {e}")
        rag_result = None
    
    # Wait before next experiment
    print("\nWaiting 30s before next experiment...")
    time.sleep(30)
    
    # =========================================================================
    # Experiment 4: Context Engineering (HARDER)
    # - 15 actions (reduced from 20 to avoid rate limits)
    # - 250 words per action
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 4: Context Engineering (HARD MODE)")
    print("  - 15 sequential actions")
    print("  - 250 words per action output")
    print("  - 1500 token limit for compression")
    print("="*70)
    
    try:
        eng_exp = ContextEngineeringExperiment_Delayed(
            config=config,
            verbose=True,
            num_actions=15,
            action_output_words=250,
            max_context_tokens=1500,
        )
        eng_result = eng_exp.run(num_trials=3, output_dir=output_dir)
        print(f"\n>>> RESULT: {eng_result.analysis.get('main_finding', 'N/A')}")
    except Exception as e:
        print(f"\n>>> ERROR in Experiment 4: {e}")
        eng_result = None
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("ALL HARD EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    if needle_result:
        print("\n1. NEEDLE IN HAYSTACK:")
        accuracies = needle_result.analysis.get('accuracies', {})
        for pos, acc in accuracies.items():
            print(f"   {pos.capitalize()}: {acc:.1%}")
    
    if size_result:
        print("\n2. CONTEXT SIZE IMPACT:")
        finding = size_result.analysis.get('main_finding', 'N/A')
        print(f"   {finding[:100]}...")
    
    if rag_result:
        print("\n3. RAG IMPACT:")
        metrics = rag_result.analysis.get('metrics', {})
        print(f"   Full Context Accuracy: {metrics.get('full', {}).get('accuracy', 0):.1%}")
        print(f"   RAG Accuracy: {metrics.get('rag', {}).get('accuracy', 0):.1%}")
    
    if eng_result:
        print("\n4. CONTEXT ENGINEERING:")
        eng_metrics = eng_result.analysis.get('metrics', {})
        for strategy, data in eng_metrics.items():
            print(f"   {strategy.capitalize()}: {data.get('accuracy', 0):.1%}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
