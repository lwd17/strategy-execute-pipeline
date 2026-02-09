#!/bin/bash
# Run all benchmark tests sequentially

set -e

echo "============================================================"
echo "Running All Benchmark Tests"
echo "============================================================"

cd "$(dirname "$0")"

# Check prerequisites
echo ""
echo "[0/3] Checking prerequisites..."

if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "✗ vLLM server is not running!"
    echo "  Please start it first: ../start_vllm_multi_gpu.sh"
    exit 1
fi
echo "✓ vLLM server is running"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "✗ OPENAI_API_KEY is not set!"
    exit 1
fi
echo "✓ OpenAI API key is set"

if [ ! -f "strategy_kg.pkl" ]; then
    echo "✗ Knowledge graph not built!"
    echo "  Run: bash scripts/build_kg.sh"
    exit 1
fi
echo "✓ Knowledge graph exists"

# Run tests
echo ""
echo "[1/3] Running AIME 2025 test..."
bash scripts/run_aime25.sh

echo ""
echo "[2/3] Running APEX test..."
bash scripts/run_apex.sh

echo ""
echo "[3/3] Running HMReasoningBench test..."
bash scripts/run_hmreasoning.sh

# Summary
echo ""
echo "============================================================"
echo "All Tests Completed!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - AIME 2025: outputs/aime25_results.json"
echo "  - APEX: outputs/apex_results.json"
echo "  - HMReasoningBench: outputs/hmreasoning_results.json"
echo ""
echo "To view statistics:"
echo "  cat outputs/aime25_results.json | jq '.statistics'"
echo "  cat outputs/apex_results.json | jq '.statistics'"
echo "  cat outputs/hmreasoning_results.json | jq '.statistics'"
echo "============================================================"
