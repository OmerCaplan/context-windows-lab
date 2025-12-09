# Context Windows Lab

**Experimental Toolkit for LLM Context Window Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for experimenting with LLM context windows, demonstrating key phenomena and comparing context management strategies.

**Course:** LLMs and Multi-Agent Systems  
**Team:** OmerAndYogever  
**Assignment:** 5 - Context Windows in Practice

---

## 🎯 Key Findings

Our experiments with Claude Haiku revealed several important insights:

| Experiment | Finding | Significance |
|------------|---------|--------------|
| **Needle in Haystack** | Claude Haiku maintains 100% accuracy even with 30 documents (~19K tokens) | Modern models may not exhibit traditional "Lost in the Middle" effects |
| **Context Size** | Latency increases 10x (1.1s → 10.6s) as context grows | Linear relationship between context size and response time |
| **RAG Impact** | 92% token savings, 97% latency reduction | RAG provides massive efficiency gains with no accuracy loss |
| **Context Engineering** | SELECT strategy: 75.6% accuracy vs 100% for others | **Statistically significant** (p=0.0007, η²=0.866) |

### 📊 Statistical Highlights

- **Context Engineering ANOVA:** F=17.29, p=0.0007 (highly significant)
- **Effect Size:** η²=0.866 (large effect)
- **RAG Token Savings:** 21,401 tokens per query (92% reduction)
- **RAG Latency Reduction:** 27.97 seconds faster (97% improvement)

---

## 📁 Project Structure

```
context-windows-lab/
├── src/
│   └── context_windows_lab/
│       ├── __init__.py              # Package exports
│       ├── config.py                # Configuration management
│       ├── cli.py                   # Command-line interface
│       ├── experiments/
│       │   ├── base.py              # Base experiment class
│       │   ├── needle_haystack.py   # Experiment 1
│       │   ├── context_size.py      # Experiment 2
│       │   ├── rag_impact.py        # Experiment 3
│       │   └── context_engineering.py # Experiment 4
│       └── utils/
│           ├── document_generator.py
│           ├── llm_client.py
│           ├── token_counter.py
│           ├── visualization.py
│           └── statistics.py
├── tests/
│   └── test_unit.py                 # Unit tests
├── notebooks/
│   ├── experiments.ipynb            # Interactive experiment runner
│   └── final_analysis.ipynb         # Results analysis & visualization
├── docs/
│   └── PRD.md                       # Product Requirements
├── outputs_hard/                    # Experiment results
├── pyproject.toml
├── .env.example
├── run_hard_experiments_v2.py       # Main experiment runner
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/OmerAndYogever/context-windows-lab.git
cd context-windows-lab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit and add your API key
nano .env
```

Your `.env` file should contain:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
CLAUDE_MODEL=claude-3-haiku-20240307
```

### Running Experiments

**Main Method: Python Script (Recommended)**

The main experiment runner includes rate limiting to avoid API errors:

```bash
python run_hard_experiments_v2.py
```

This script runs all 4 experiments with harder parameters:
- Needle in Haystack: 30 documents, 500 words each
- Context Size: 10, 30, 60 documents
- RAG Impact: 40 documents
- Context Engineering: 15 actions

Results are saved to `outputs_hard/` directory.

**Alternative: Python API**
```python
from context_windows_lab import (
    Config,
    NeedleInHaystackExperiment,
    ContextSizeExperiment,
    RAGImpactExperiment,
    ContextEngineeringExperiment,
)

config = Config.from_env()

# Run Experiment 4 (most interesting results)
exp = ContextEngineeringExperiment(config=config, num_actions=15)
result = exp.run(num_trials=3)
print(result.analysis['main_finding'])
```

**Alternative: Jupyter Notebook**
```bash
# Run experiments interactively
jupyter notebook notebooks/experiments.ipynb

# Analyze results (after running experiments)
jupyter notebook notebooks/final_analysis.ipynb
```

The `final_analysis.ipynb` notebook loads results from `outputs_hard/` and provides:
- Detailed analysis of each experiment
- Statistical interpretation
- Publication-quality visualizations
- Summary dashboard with recommendations

---

## 📈 Experiment Details

### Experiment 1: Needle in Haystack

Tests the "Lost in the Middle" phenomenon by placing critical facts at different positions.

**Parameters (Hard Mode):**
- Documents: 30
- Words per document: 500
- Total tokens: ~19,000

**Result:** Claude Haiku achieved 100% accuracy across all positions (start, middle, end), suggesting modern models have improved attention mechanisms that mitigate the traditional "Lost in the Middle" effect.

### Experiment 2: Context Size Impact

Measures accuracy and latency as context window size increases.

**Parameters:**
- Document counts: 10, 30 documents
- Words per document: 500

**Results:**
| Documents | Tokens | Accuracy | Latency |
|-----------|--------|----------|---------|
| 10 | 5,778 | 100% | 1.10s |
| 30 | 17,354 | 100% | 10.64s |

**Finding:** While accuracy remains constant, latency scales linearly with context size.

### Experiment 3: RAG Impact

Compares full-context retrieval with RAG (Retrieval Augmented Generation).

**Parameters:**
- Total documents: 40
- RAG retrieves: 3 documents

**Results:**
| Metric | Full Context | RAG | Improvement |
|--------|-------------|-----|-------------|
| Accuracy | 100% | 100% | - |
| Latency | 28.65s | 0.68s | **97% faster** |
| Tokens | 23,138 | 1,738 | **92% savings** |

**Finding:** RAG provides massive efficiency gains with no accuracy penalty.

### Experiment 4: Context Engineering Strategies ⭐

Tests four strategies for managing context in multi-step agent workflows.

**Strategies:**
- **BASELINE:** Keep all history
- **SELECT:** Keep only recent k items
- **COMPRESS:** Summarize when too long
- **WRITE:** Extract key facts to scratchpad

**Results (15 actions, 3 trials):**

| Strategy | Accuracy | Latency | Tokens |
|----------|----------|---------|--------|
| Baseline | 100% | 1.66s | 2,634 |
| Select | **75.6%** | 1.50s | 950 |
| Compress | 100% | 1.51s | 809 |
| Write | 100% | 1.38s | 518 |

**Statistical Analysis:**
- ANOVA: F=17.29, **p=0.0007** (highly significant)
- Effect size: η²=0.866 (large effect)

**Finding:** The SELECT strategy significantly degrades performance by discarding important context. COMPRESS and WRITE maintain accuracy while reducing token usage.

---

## 🔬 Statistical Methods

The toolkit includes comprehensive statistical analysis:

- **Descriptive Statistics:** Mean, SD, median, range, SEM
- **Confidence Intervals:** 95% CI for all metrics
- **Hypothesis Testing:**
  - Independent samples t-test (pairwise comparisons)
  - One-way ANOVA (multi-group comparisons)
  - Pearson correlation (size-performance relationships)
- **Effect Sizes:**
  - Cohen's d (t-tests)
  - η² eta-squared (ANOVA)
  - R² (correlations)

---

## 📊 Visualization

All experiments generate publication-quality plots:

- Bar charts for accuracy comparisons
- Line plots for size-performance relationships
- Multi-panel comparisons for strategies
- Error bars showing standard deviation

Output files are saved to `outputs_hard/` directory.

---

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `CLAUDE_MODEL` | Model to use | `claude-3-haiku-20240307` |
| `MAX_TOKENS` | Max response tokens | `4096` |
| `NUM_EXPERIMENT_RUNS` | Trials per experiment | `5` |
| `RANDOM_SEED` | Seed for reproducibility | `42` |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_unit.py
```

---

## 📝 Files Generated

After running experiments:

```
outputs_hard/
├── needle_in_haystack_results.json
├── needle_haystack_accuracy.png
├── context_size_impact_results.json
├── context_size_impact.png
├── rag_impact_results.json
├── rag_comparison.png
├── context_engineering_results.json
└── context_engineering_comparison.png
```

---

## 🔧 Troubleshooting

### Rate Limiting (Error 429)
The `run_hard_experiments_v2.py` script includes automatic delays to avoid rate limits. If you still encounter issues, wait 1-2 minutes before retrying.

### API Key Not Found
```bash
# Ensure .env file exists and contains your key
cat .env
# Should show: ANTHROPIC_API_KEY=sk-ant-...
```

### Module Not Found
```bash
# Reinstall in development mode
pip install -e .
```

---

## 📚 References

1. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts"
2. Anthropic Claude Documentation: https://docs.anthropic.com
3. LangChain RAG Documentation: https://python.langchain.com

---

## 🤖 AI Interaction

This project was developed with assistance from Claude (Anthropic). AI was used for:
- Code architecture and implementation
- Documentation writing
- Statistical analysis design
- Debugging and optimization

All AI-generated code was reviewed, tested, and validated by the team.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👥 Team

**OmerAndYogever**
- Omer Caplan (ID: 208753665)
- Yogev Cuperman (ID: 207540550)

Course: LLMs and Multi-Agent Systems  
Date: December 2025
