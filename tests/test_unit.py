"""
Unit tests for Context Windows Lab.

Tests utility functions and experiment components without requiring API access.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_windows_lab.config import Config
from context_windows_lab.utils.document_generator import (
    DocumentGenerator, 
    FactPosition, 
    GeneratedDocument
)
from context_windows_lab.utils.token_counter import TokenCounter, count_tokens
from context_windows_lab.utils.statistics import StatisticalAnalyzer, StatisticalResult


class TestConfig:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.max_tokens == 4096
        assert config.num_runs == 5
        assert config.random_seed == 42
    
    def test_validate_without_api_key(self):
        """Test validation fails without API key."""
        config = Config()
        config.anthropic_api_key = ""
        errors = config.validate()
        assert len(errors) > 0
        assert any("API_KEY" in e for e in errors)
    
    def test_has_api_key_property(self):
        """Test has_api_key property."""
        config = Config()
        config.anthropic_api_key = ""
        assert not config.has_api_key
        
        config.anthropic_api_key = "test-key"
        assert config.has_api_key


class TestDocumentGenerator:
    """Tests for document generation."""
    
    def test_generate_filler_text(self):
        """Test filler text generation."""
        gen = DocumentGenerator(seed=42)
        text = gen.generate_filler_text(target_words=100)
        
        # Should have approximately 100 words
        word_count = len(text.split())
        assert word_count >= 100
    
    def test_embed_fact_start(self):
        """Test embedding fact at start."""
        gen = DocumentGenerator(seed=42)
        text = "This is some text."
        fact = "Important fact."
        
        result = gen.embed_fact(text, fact, FactPosition.START)
        assert result.startswith(fact)
    
    def test_embed_fact_end(self):
        """Test embedding fact at end."""
        gen = DocumentGenerator(seed=42)
        text = "This is some text."
        fact = "Important fact."
        
        result = gen.embed_fact(text, fact, FactPosition.END)
        assert result.endswith(fact)
    
    def test_generate_document(self):
        """Test single document generation."""
        gen = DocumentGenerator(seed=42)
        doc = gen.generate_document(
            document_id=1,
            words_per_doc=200,
            fact_position=FactPosition.MIDDLE,
            fact_index=0,
        )
        
        assert isinstance(doc, GeneratedDocument)
        assert doc.document_id == 1
        assert doc.fact_position == FactPosition.MIDDLE
        assert doc.word_count > 0
    
    def test_generate_document_set(self):
        """Test document set generation."""
        gen = DocumentGenerator(seed=42)
        docs = gen.generate_document_set(num_documents=5)
        
        assert len(docs) == 5
        for i, doc in enumerate(docs):
            assert doc.document_id == i
    
    def test_create_needle_haystack_context(self):
        """Test needle in haystack context creation."""
        gen = DocumentGenerator(seed=42)
        context, query, expected = gen.create_needle_haystack_context(
            num_documents=5,
            needle_position=FactPosition.MIDDLE,
        )
        
        assert len(context) > 0
        assert len(query) > 0
        assert len(expected) > 0
        assert "CEO" in query or "company" in query.lower()
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same output."""
        gen1 = DocumentGenerator(seed=42)
        gen2 = DocumentGenerator(seed=42)
        
        text1 = gen1.generate_filler_text(50)
        text2 = gen2.generate_filler_text(50)
        
        assert text1 == text2


class TestTokenCounter:
    """Tests for token counting."""
    
    def test_count_tokens(self):
        """Test basic token counting."""
        counter = TokenCounter()
        text = "Hello, world! This is a test."
        count = counter.count(text)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_empty_string(self):
        """Test counting empty string."""
        counter = TokenCounter()
        assert counter.count("") == 0
    
    def test_count_documents(self):
        """Test counting multiple documents."""
        counter = TokenCounter()
        docs = ["Hello world", "Goodbye world", "Test document"]
        result = counter.count_documents(docs)
        
        assert "total" in result
        assert "per_document" in result
        assert "average" in result
        assert len(result["per_document"]) == 3
    
    def test_truncate_to_tokens(self):
        """Test token truncation."""
        counter = TokenCounter()
        text = "This is a long text that should be truncated. " * 10
        truncated = counter.truncate_to_tokens(text, max_tokens=10)
        
        assert counter.count(truncated) <= 10
    
    def test_split_by_tokens(self):
        """Test splitting by tokens."""
        counter = TokenCounter()
        text = "Word " * 100
        chunks = counter.split_by_tokens(text, chunk_size=20)
        
        assert len(chunks) > 1
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert counter.count(chunk) <= 20
    
    def test_convenience_function(self):
        """Test the convenience count_tokens function."""
        result = count_tokens("Hello world")
        assert result > 0


class TestStatisticalAnalyzer:
    """Tests for statistical analysis."""
    
    def test_descriptive_stats(self):
        """Test descriptive statistics calculation."""
        analyzer = StatisticalAnalyzer()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = analyzer.descriptive_stats(data)
        
        assert stats["n"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = analyzer.confidence_interval(data)
        
        assert lower < 3.0 < upper
    
    def test_independent_t_test(self):
        """Test independent t-test."""
        analyzer = StatisticalAnalyzer()
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        
        result = analyzer.independent_t_test(group1, group2)
        
        assert isinstance(result, StatisticalResult)
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant
    
    def test_one_way_anova(self):
        """Test one-way ANOVA."""
        analyzer = StatisticalAnalyzer()
        groups = {
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
            "C": [7.0, 8.0, 9.0],
        }
        
        result = analyzer.one_way_anova(groups)
        
        assert isinstance(result, StatisticalResult)
        assert result.test_name == "One-way ANOVA"
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        analyzer = StatisticalAnalyzer()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect positive correlation
        
        result = analyzer.correlation_analysis(x, y)
        
        assert abs(result.statistic - 1.0) < 0.001  # r should be ~1
    
    def test_summarize_experiment(self):
        """Test experiment summarization."""
        analyzer = StatisticalAnalyzer()
        conditions = {
            "control": [0.5, 0.6, 0.7],
            "treatment": [0.8, 0.9, 0.85],
        }
        
        summary = analyzer.summarize_experiment(conditions)
        
        assert "conditions" in summary
        assert "control" in summary["conditions"]
        assert "treatment" in summary["conditions"]


class TestExperimentBase:
    """Tests for experiment base functionality."""
    
    def test_experiment_result_to_dict(self):
        """Test ExperimentResult serialization."""
        from context_windows_lab.experiments.base import ExperimentResult
        
        result = ExperimentResult(
            experiment_name="Test",
            raw_results={"test": 1},
            analysis={"finding": "test"},
        )
        
        d = result.to_dict()
        assert d["experiment_name"] == "Test"
        assert d["raw_results"]["test"] == 1
    
    def test_experiment_result_save_load(self, tmp_path):
        """Test saving and loading results."""
        from context_windows_lab.experiments.base import ExperimentResult
        
        result = ExperimentResult(
            experiment_name="Test",
            raw_results={"test": 1},
        )
        
        path = tmp_path / "test_result.json"
        result.save(path)
        
        loaded = ExperimentResult.load(path)
        assert loaded.experiment_name == "Test"
        assert loaded.raw_results["test"] == 1


class TestIntegration:
    """Integration tests (mock API calls)."""
    
    @patch("context_windows_lab.utils.llm_client.Anthropic")
    def test_llm_client_query_mocked(self, mock_anthropic):
        """Test LLM client with mocked API."""
        from context_windows_lab.utils.llm_client import LLMClient
        from context_windows_lab.config import Config
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        config = Config()
        config.anthropic_api_key = "test-key"
        
        client = LLMClient(config)
        response = client.query("context", "question")
        
        assert response.content == "Test response"
        assert response.input_tokens == 100
    
    def test_evaluate_response_correct(self):
        """Test response evaluation - correct answer."""
        from context_windows_lab.utils.llm_client import LLMClient
        from context_windows_lab.config import Config
        
        config = Config()
        config.anthropic_api_key = ""  # No API key, but we can still test evaluation
        
        client = LLMClient(config)
        
        is_correct, confidence = client.evaluate_response(
            "The CEO is David Cohen.",
            "David Cohen"
        )
        
        assert is_correct
        assert confidence > 0
    
    def test_evaluate_response_incorrect(self):
        """Test response evaluation - incorrect answer."""
        from context_windows_lab.utils.llm_client import LLMClient
        from context_windows_lab.config import Config
        
        config = Config()
        client = LLMClient(config)
        
        is_correct, confidence = client.evaluate_response(
            "The CEO is John Smith.",
            "David Cohen"
        )
        
        assert not is_correct
        assert confidence == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
