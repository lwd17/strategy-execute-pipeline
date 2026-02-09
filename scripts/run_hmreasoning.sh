#!/bin/bash
# Run HMReasoningBench test with train/test split
# This ensures no data leakage

set -e

echo "============================================================"
echo "Running HMReasoningBench with Train/Test Split"
echo "============================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check if vLLM server is running
echo ""
echo "[1/6] Checking vLLM server..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "✗ vLLM server is not running!"
    echo ""
    echo "Please start the vLLM server first:"
    echo "  cd .."
    echo "  ./start_vllm_multi_gpu.sh"
    echo ""
    exit 1
fi
echo "✓ vLLM server is running"

# Check if OpenAI API key is set
echo ""
echo "[2/6] Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "✗ OPENAI_API_KEY is not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    exit 1
fi
echo "✓ OpenAI API key is set"

# Split dataset (if not already done)
echo ""
echo "[3/6] Splitting dataset into train/test..."
if [ -f "strategy_dataset_train.json" ] && [ -f "strategy_dataset_test.json" ]; then
    echo "✓ Train/test split already exists, skipping..."
else
    echo "  Running split_dataset_unique.py..."
    python src/split_dataset_unique.py
fi

# Build KG from train set only (if not already done)
echo ""
echo "[4/6] Building knowledge graph from train set..."
if [ -f "strategy_kg_train.pkl" ] && [ -f "gnn_model_train.pth" ]; then
    echo "✓ Train set KG already exists, skipping..."
    echo "  (Delete strategy_kg_train.pkl to force rebuild)"
else
    echo "  Running build_kg_train_only.py..."
    python src/build_kg_train_only.py
fi

# Verify no data leakage
echo ""
echo "[5/6] Verifying no data leakage..."
python tests/verify_no_leakage.py

# Run test on test set
echo ""
echo "[6/6] Running HMReasoningBench test..."
echo "  Model: ${VLLM_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
echo "  Judge: ${OPENAI_MODEL_JUDGE:-gpt-4o}"
echo "  Test set: 103 unique problems"
echo "  KG built from: 4832 train problems"
echo ""

python tests/test_hmreasoning.py

echo ""
echo "============================================================"
echo "✓ HMReasoningBench test completed!"
echo "Results saved to: outputs/hmreasoning_results.json"
echo "============================================================"
echo ""
echo "Summary:"
echo "  - Train set: strategy_dataset_train.json (4832 problems)"
echo "  - Test set: strategy_dataset_test.json (103 problems)"
echo "  - KG files: strategy_kg_train.pkl, gnn_model_train.pth, etc."
echo "  - No data leakage verified ✓"
echo "============================================================"