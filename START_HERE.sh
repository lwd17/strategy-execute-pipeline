#!/bin/bash
# Complete End-to-End Workflow
# This script guides you through the entire pipeline

set -e

echo "============================================================"
echo "   Math Strategy Pipeline - Complete Workflow"
echo "============================================================"
echo ""
echo "This script will guide you through:"
echo "  1. Starting vLLM server (in background)"
echo "  2. Building knowledge graph"
echo "  3. Running all benchmark tests"
echo ""
echo "Prerequisites:"
echo "  - Python dependencies installed (pip install -r requirements.txt)"
echo "  - OPENAI_API_KEY environment variable set"
echo "  - CUDA available (for GPU acceleration)"
echo ""

# Check prerequisites
echo "============================================================"
echo "Step 0: Checking Prerequisites"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 not found!"
    echo "  Please install Python 3.8 or higher"
    exit 1
fi
echo "✓ Python3 found: $(python3 --version)"

# Check OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "✗ OPENAI_API_KEY not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    exit 1
fi
echo "✓ OpenAI API key is set"

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "⚠ vLLM not installed"
    echo "  Install with: pip install vllm"
    echo "  (Will try to continue anyway)"
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=index,name --format=csv,noheader | nl
else
    echo "⚠ CUDA not available (will use CPU, slower)"
fi

echo ""
echo "Press Enter to continue, or Ctrl+C to abort..."
read -r

# Step 1: Start vLLM server in background
echo ""
echo "============================================================"
echo "Step 1: Starting vLLM Server"
echo "============================================================"
echo ""

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ vLLM server already running at http://localhost:8000"
    echo "  Skipping server start"
else
    echo "Starting vLLM server in background..."
    echo "  This will take 30-120 seconds to load the model"
    echo ""

    # Start server in background
    bash scripts/start_vllm_server.sh > logs/vllm_startup.log 2>&1 &
    VLLM_PID=$!
    echo "  Server PID: $VLLM_PID"
    echo "  Log file: logs/vllm_startup.log"
    echo ""

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    MAX_WAIT=180  # 3 minutes
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            break
        fi

        # Check if process is still alive
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "✗ Server process died!"
            echo "  Check logs: tail -f logs/vllm_startup.log"
            exit 1
        fi

        echo -n "."
        sleep 5
        WAITED=$((WAITED + 5))
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        echo "✗ Server failed to start within $MAX_WAIT seconds"
        echo "  Check logs: tail -f logs/vllm_startup.log"
        exit 1
    fi
fi

echo ""
echo "Press Enter to continue to Step 2 (Build Knowledge Graph)..."
read -r

# Step 2: Build knowledge graph
echo ""
echo "============================================================"
echo "Step 2: Building Knowledge Graph"
echo "============================================================"
echo ""

if [ -f "strategy_kg.pkl" ] && [ -f "gnn_model.pth" ]; then
    echo "Knowledge graph files already exist:"
    ls -lh strategy_kg.pkl gnn_model.pth *_emb.npy 2>/dev/null | head -5
    echo ""
    echo "Do you want to rebuild? (y/N)"
    read -r REBUILD
    if [ "$REBUILD" != "y" ] && [ "$REBUILD" != "Y" ]; then
        echo "Skipping KG rebuild"
    else
        echo "Rebuilding knowledge graph..."
        bash scripts/build_kg.sh
    fi
else
    echo "Building knowledge graph for the first time..."
    echo "This will take 10-20 minutes on GPU, 30-60 minutes on CPU"
    echo ""
    bash scripts/build_kg.sh
fi

echo ""
echo "Press Enter to continue to Step 3 (Run Tests)..."
read -r

# Step 3: Run tests
echo ""
echo "============================================================"
echo "Step 3: Running Benchmark Tests"
echo "============================================================"
echo ""
echo "Which test do you want to run?"
echo "  1) AIME 2025 only (~15-30 min)"
echo "  2) APEX only (~30-60 min)"
echo "  3) HMReasoningBench only (~30-60 min)"
echo "  4) All tests (~2-3 hours)"
echo "  5) Skip tests (exit)"
echo ""
echo -n "Enter choice (1-5): "
read -r CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Running AIME 2025..."
        bash scripts/run_aime25.sh
        ;;
    2)
        echo ""
        echo "Running APEX..."
        bash scripts/run_apex.sh
        ;;
    3)
        echo ""
        echo "Running HMReasoningBench..."
        bash scripts/run_hmreasoning.sh
        ;;
    4)
        echo ""
        echo "Running all tests..."
        bash RUN_ALL_TESTS.sh
        ;;
    5)
        echo ""
        echo "Skipping tests"
        ;;
    *)
        echo "Invalid choice, skipping tests"
        ;;
esac

# Summary
echo ""
echo "============================================================"
echo "Workflow Complete!"
echo "============================================================"
echo ""

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ vLLM server is still running at http://localhost:8000"
    echo "  To stop: pkill -f vllm.entrypoints.openai.api_server"
fi

if [ -f "strategy_kg.pkl" ]; then
    echo "✓ Knowledge graph built"
fi

if ls outputs/*.json 1> /dev/null 2>&1; then
    echo "✓ Test results available:"
    ls -lh outputs/*.json 2>/dev/null
fi

echo ""
echo "Next steps:"
echo "  - View results: cat outputs/aime25_results.json | jq '.statistics'"
echo "  - Read documentation: cat README.md"
echo "  - Run specific tests: bash scripts/run_aime25.sh"
echo ""
echo "To stop vLLM server:"
echo "  pkill -f vllm.entrypoints.openai.api_server"
echo ""
echo "============================================================"
