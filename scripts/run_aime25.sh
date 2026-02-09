#!/bin/bash
# Run AIME 2025 benchmark test

set -e

echo "============================================================"
echo "Running AIME 2025 Benchmark"
echo "============================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check if vLLM server is running
echo ""
echo "[1/3] Checking vLLM server..."
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
echo "[2/3] Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "✗ OPENAI_API_KEY is not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    exit 1
fi
echo "✓ OpenAI API key is set"

# Run test
echo ""
echo "[3/3] Running AIME 2025 test..."
echo "  Model: ${VLLM_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
echo "  Judge: ${OPENAI_MODEL_JUDGE:-gpt-4o}"
echo ""

python tests/test_aime25_accuracy.py

echo ""
echo "============================================================"
echo "✓ AIME 2025 test completed!"
echo "Results saved to: outputs/aime25_results.json"
echo "============================================================"