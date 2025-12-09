"""
Statistical Analysis utilities for Context Windows Lab.

Provides statistical tests and analysis for experiment results.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """
    Result of a statistical analysis.
    
    Attributes:
        test_name: Name of the statistical test performed
        statistic: The test statistic value
        p_value: The p-value of the test
        effect_size: Effect size measure (if applicable)
        is_significant: Whether result is statistically significant (p < 0.05)
        interpretation: Human-readable interpretation
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    is_significant: bool = False
    interpretation: str = ""
    
    def __post_init__(self):
        self.is_significant = self.p_value < 0.05


class StatisticalAnalyzer:
    """
    Performs statistical analysis on experiment results.
    
    Provides methods for:
    - Descriptive statistics
    - Hypothesis testing (t-tests, ANOVA)
    - Effect size calculations
    - Confidence intervals
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
    
    def descriptive_stats(self, data: list[float]) -> dict:
        """
        Calculate descriptive statistics for a dataset.
        
        Args:
            data: List of numeric values
            
        Returns:
            Dictionary with mean, std, median, min, max, etc.
        """
        arr = np.array(data)
        
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "variance": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "sem": float(stats.sem(arr)) if len(arr) > 1 else 0.0,
        }
    
    def confidence_interval(
        self, 
        data: list[float], 
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data: List of numeric values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        arr = np.array(data)
        n = len(arr)
        
        if n < 2:
            mean = np.mean(arr)
            return (mean, mean)
        
        mean = np.mean(arr)
        sem = stats.sem(arr)
        
        # Use t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * sem
        
        return (mean - margin, mean + margin)
    
    def independent_t_test(
        self,
        group1: list[float],
        group2: list[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
    ) -> StatisticalResult:
        """
        Perform independent samples t-test.
        
        Args:
            group1: Data for first group
            group2: Data for second group
            group1_name: Name for first group
            group2_name: Name for second group
            
        Returns:
            StatisticalResult with test outcomes
        """
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1, ddof=1) + 
             (len(group2) - 1) * np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        effect_interpretation = "negligible"
        if abs(cohens_d) >= 0.8:
            effect_interpretation = "large"
        elif abs(cohens_d) >= 0.5:
            effect_interpretation = "medium"
        elif abs(cohens_d) >= 0.2:
            effect_interpretation = "small"
        
        interpretation = (
            f"Comparing {group1_name} (M={np.mean(group1):.3f}) vs "
            f"{group2_name} (M={np.mean(group2):.3f}): "
            f"t={t_stat:.3f}, p={p_value:.4f}. "
            f"Effect size (Cohen's d): {cohens_d:.3f} ({effect_interpretation}). "
            f"{'Statistically significant' if p_value < self.alpha else 'Not statistically significant'}."
        )
        
        return StatisticalResult(
            test_name="Independent Samples t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            interpretation=interpretation,
        )
    
    def paired_t_test(
        self,
        before: list[float],
        after: list[float],
    ) -> StatisticalResult:
        """
        Perform paired samples t-test.
        
        Args:
            before: Pre-treatment measurements
            after: Post-treatment measurements
            
        Returns:
            StatisticalResult with test outcomes
        """
        t_stat, p_value = stats.ttest_rel(before, after)
        
        # Calculate effect size (Cohen's d for paired samples)
        diff = np.array(after) - np.array(before)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        interpretation = (
            f"Paired comparison: Before (M={np.mean(before):.3f}) vs "
            f"After (M={np.mean(after):.3f}): "
            f"t={t_stat:.3f}, p={p_value:.4f}. "
            f"Effect size: {cohens_d:.3f}. "
            f"{'Significant difference' if p_value < self.alpha else 'No significant difference'}."
        )
        
        return StatisticalResult(
            test_name="Paired Samples t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            interpretation=interpretation,
        )
    
    def one_way_anova(
        self,
        groups: dict[str, list[float]],
    ) -> StatisticalResult:
        """
        Perform one-way ANOVA.
        
        Args:
            groups: Dictionary mapping group names to data lists
            
        Returns:
            StatisticalResult with test outcomes
        """
        group_data = list(groups.values())
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate eta-squared effect size
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 
            for g in group_data
        )
        ss_total = sum((x - grand_mean) ** 2 for x in all_data)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        group_summary = ", ".join(
            f"{name}: M={np.mean(data):.3f}" 
            for name, data in groups.items()
        )
        
        interpretation = (
            f"ANOVA across {len(groups)} groups ({group_summary}): "
            f"F={f_stat:.3f}, p={p_value:.4f}. "
            f"Effect size (η²): {eta_squared:.3f}. "
            f"{'Significant differences exist' if p_value < self.alpha else 'No significant differences'}."
        )
        
        return StatisticalResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            effect_size=eta_squared,
            interpretation=interpretation,
        )
    
    def correlation_analysis(
        self,
        x: list[float],
        y: list[float],
        x_name: str = "X",
        y_name: str = "Y",
    ) -> StatisticalResult:
        """
        Perform Pearson correlation analysis.
        
        Args:
            x: First variable
            y: Second variable
            x_name: Name of first variable
            y_name: Name of second variable
            
        Returns:
            StatisticalResult with correlation outcomes
        """
        r, p_value = stats.pearsonr(x, y)
        
        # Interpret correlation strength
        strength = "negligible"
        if abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.4:
            strength = "moderate"
        elif abs(r) >= 0.2:
            strength = "weak"
        
        direction = "positive" if r > 0 else "negative"
        
        interpretation = (
            f"Correlation between {x_name} and {y_name}: "
            f"r={r:.3f}, p={p_value:.4f}. "
            f"This is a {strength} {direction} correlation. "
            f"{'Statistically significant' if p_value < self.alpha else 'Not statistically significant'}."
        )
        
        return StatisticalResult(
            test_name="Pearson Correlation",
            statistic=r,
            p_value=p_value,
            effect_size=r ** 2,  # R-squared
            interpretation=interpretation,
        )
    
    def summarize_experiment(
        self,
        conditions: dict[str, list[float]],
        metric_name: str = "accuracy",
    ) -> dict:
        """
        Generate a comprehensive summary of experiment results.
        
        Args:
            conditions: Dict mapping condition names to result lists
            metric_name: Name of the metric being measured
            
        Returns:
            Dictionary with descriptive stats, tests, and interpretations
        """
        summary = {
            "metric": metric_name,
            "conditions": {},
            "statistical_tests": [],
        }
        
        # Descriptive stats for each condition
        for name, data in conditions.items():
            summary["conditions"][name] = self.descriptive_stats(data)
            ci = self.confidence_interval(data)
            summary["conditions"][name]["ci_95"] = {"lower": ci[0], "upper": ci[1]}
        
        # ANOVA if more than 2 conditions
        if len(conditions) > 2:
            anova_result = self.one_way_anova(conditions)
            summary["statistical_tests"].append({
                "test": anova_result.test_name,
                "statistic": anova_result.statistic,
                "p_value": anova_result.p_value,
                "effect_size": anova_result.effect_size,
                "significant": anova_result.is_significant,
                "interpretation": anova_result.interpretation,
            })
        elif len(conditions) == 2:
            # T-test for 2 conditions
            names = list(conditions.keys())
            t_result = self.independent_t_test(
                conditions[names[0]], 
                conditions[names[1]],
                names[0],
                names[1],
            )
            summary["statistical_tests"].append({
                "test": t_result.test_name,
                "statistic": t_result.statistic,
                "p_value": t_result.p_value,
                "effect_size": t_result.effect_size,
                "significant": t_result.is_significant,
                "interpretation": t_result.interpretation,
            })
        
        return summary
