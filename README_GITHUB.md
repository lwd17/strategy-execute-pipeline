# Math Strategy Pipeline

A complete end-to-end pipeline for mathematical problem-solving using strategy-guided retrieval and large language models.

## 🎯 Features

- **Three-Path Retrieval System**: Semantic, structural, and template-based strategy retrieval
- **Pre-trained Models**: Ready-to-use knowledge graph with 4,932 problems and 43,472 strategies
- **Self-contained**: Includes vLLM server startup, testing, and evaluation
- **Benchmark Support**: AIME 2025, APEX, and HMReasoningBench datasets
- **Baseline Performance**: 73.33% accuracy on AIME 2025 (22/30 problems)

## 📦 Quick Start

See [START.md](START.md) for detailed 3-step setup guide.

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# Download pre-trained models (see releases)
# Download dataset (see data/README.md)
```

### 1-Minute Test

```bash
# Check setup
python check_kg.py
python verify_setup.py

# Start vLLM server
bash scripts/start_vllm_server.sh

# Run test (in new terminal)
export VLLM_MODEL="Qwen/Qwen3-8B"
export OPENAI_API_KEY="sk-..."
python tests/test_aime25_accuracy.py
```

## 📚 Documentation

- **[START.md](START.md)**: Quick start guide (3 steps)
- **[README.md](README.md)**: Full documentation
- **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)**: Technical details

## 🏗️ Architecture

### Three-Path Retrieval

1. **Path A**: Similar problems → retrieve their strategies
2. **Path B**: Similar problems → structural embeddings → templates → strategies  
3. **Path C**: Direct semantic matching between query and strategies

Final ranking combines all three paths with diversity control.

### Strategy Templates

30 predefined templates across 6 categories:
- Coordinate Methods (5 templates)
- Geometric Methods (6 templates)
- Number Theory (6 templates)
- Combinatorial (5 templates)
- Algebraic (5 templates)
- Other Methods (3 templates)

## 📊 Pre-trained Models

Available in [Releases](../../releases):

| File | Size | Description |
|------|------|-------------|
| `strategy_kg.pkl` | 41 MB | Knowledge graph (4,932 problems, 43,472 strategies) |
| `gnn_model.pth` | 1.5 MB | Trained Graph Neural Network |
| `strategy_classifier.pkl` | 64 MB | Strategy suitability classifier |
| `*_emb.npy` | ~140 MB | Pre-computed embeddings (6 files) |

**Total**: ~200 MB

## 🎓 Training from Scratch

Optional - you can train your own knowledge graph:

```bash
# Prepare data (see data/README.md)
# Then run:
python src/build_kg.py
```

Training time: ~10-20 minutes on GPU.

## 🧪 Testing

```bash
# AIME 2025 (30 problems)
bash scripts/run_aime25.sh

# APEX
bash scripts/run_apex.sh

# HMReasoningBench
bash scripts/run_hmreasoning.sh
```

## 📈 Benchmark Results

| Dataset | Accuracy | Correct/Total |
|---------|----------|---------------|
| AIME 2025 | 73.33% | 22/30 |
| APEX | TBD | TBD |
| HMReasoningBench | TBD | TBD |

## 🔧 Supported Models

- Qwen/Qwen3-8B (recommended)
- Qwen/Qwen3-14B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{math_strategy_pipeline,
  title={Math Strategy Pipeline: Strategy-Guided Mathematical Problem Solving},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/math_strategy_pipeline}
}
```

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or issues, please open a GitHub issue.
