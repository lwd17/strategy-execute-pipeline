#!/bin/bash

# vLLM Server Startup Script
# This script starts a vLLM server with configurable GPU settings

echo "============================================================"
echo "Starting vLLM Server"
echo "============================================================"

# Change to parent directory for log file
cd "$(dirname "$0")/.."

# Kill existing process
echo ""
echo "Checking for existing vLLM processes..."
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "Found existing vLLM process, killing..."
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    sleep 2
    echo "✓ Killed existing process"
else
    echo "✓ No existing process found"
fi

# Model selection (if not provided via environment variable)
if [ -z "$MODEL_NAME" ]; then
    echo ""
    echo "============================================================"
    echo "Model Selection"
    echo "============================================================"
    echo ""
    echo "Available models:"
    echo "  1) Qwen/Qwen3-8B"
    echo "  2) Qwen/Qwen3-14B"
    echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    echo "  4) Custom (enter model name)"
    echo ""
    echo -n "Select model (1-4) [default: 1]: "
    read -r MODEL_CHOICE

    case ${MODEL_CHOICE:-1} in
        1)
            MODEL_NAME="Qwen/Qwen3-8B"
            ;;
        2)
            MODEL_NAME="Qwen/Qwen3-14B"
            ;;
        3)
            MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        4)
            echo -n "Enter model name: "
            read -r MODEL_NAME
            ;;
        *)
            echo "Invalid choice, using default: Qwen/Qwen3-8B"
            MODEL_NAME="Qwen/Qwen3-8B"
            ;;
    esac
fi

# GPU selection (if not provided via environment variable)
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    echo ""
    echo "============================================================"
    echo "GPU Configuration"
    echo "============================================================"
    echo ""

    # Check available GPUs
    if ! command -v nvidia-smi &> /dev/null; then
        echo "✗ nvidia-smi not found. GPU not available?"
        echo "  Please check your CUDA installation"
        exit 1
    fi

    # Count available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $NUM_GPUS GPU(s)"
    echo ""

    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | nl
    echo ""

    echo "How many GPUs to use for tensor parallelism?"
    echo "  1) 1 GPU"
    echo "  2) 2 GPUs"
    echo "  3) 4 GPUs"
    echo "  4) 8 GPUs"
    echo ""
    echo -n "Select (1-4) [default: 1]: "
    read -r GPU_CHOICE

    case ${GPU_CHOICE:-1} in
        1)
            TENSOR_PARALLEL_SIZE=1
            ;;
        2)
            TENSOR_PARALLEL_SIZE=2
            ;;
        3)
            TENSOR_PARALLEL_SIZE=4
            ;;
        4)
            TENSOR_PARALLEL_SIZE=8
            ;;
        *)
            echo "Invalid choice, using 1 GPU"
            TENSOR_PARALLEL_SIZE=1
            ;;
    esac
fi

# Set default values for other parameters
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-35000}

echo ""
echo "============================================================"
echo "Final Configuration"
echo "============================================================"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPU(s)"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo ""

# Automatically select GPUs with most free memory
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    echo "Selecting $TENSOR_PARALLEL_SIZE GPUs with most free memory..."
    SELECTED_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                    sort -t, -k2 -nr | \
                    head -n $TENSOR_PARALLEL_SIZE | \
                    cut -d, -f1 | \
                    tr '\n' ',' | \
                    sed 's/,$//')

    if [ -z "$SELECTED_GPUS" ]; then
        echo "✗ Failed to select GPUs automatically"
        echo "  Using default GPUs: 0,1,2,3 (first $TENSOR_PARALLEL_SIZE)"
        SELECTED_GPUS=$(seq -s, 0 $((TENSOR_PARALLEL_SIZE-1)))
    fi

    echo "✓ Selected GPUs: $SELECTED_GPUS"
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
else
    echo "Using single GPU mode"
    # For single GPU, use the one with most free memory
    SELECTED_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                   sort -t, -k2 -nr | \
                   head -n 1 | \
                   cut -d, -f1)
    echo "✓ Selected GPU: $SELECTED_GPU"
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
fi

# Create logs directory
mkdir -p logs

LOG_FILE="logs/vllm_server.log"

echo ""
echo "============================================================"
echo "Starting vLLM Service"
echo "============================================================"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "Wait for 'Application startup complete' message..."
echo "Then the server will be ready at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Build vLLM command
VLLM_CMD="vllm serve \"$MODEL_NAME\" \
    --host \"$HOST\" \
    --port \"$PORT\" \
    --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
    --max-model-len \"$MAX_MODEL_LEN\" \
    --max-num-seqs 256 \
    --trust-remote-code \
    --disable-log-requests"

# Add tensor parallel size if > 1
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    VLLM_CMD="$VLLM_CMD --tensor-parallel-size \"$TENSOR_PARALLEL_SIZE\""
fi

# Add reasoning parser for DeepSeek and Qwen3 models
if [[ "$MODEL_NAME" == *"DeepSeek"* ]] || [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
    VLLM_CMD="$VLLM_CMD --reasoning-parser qwen3"
    echo "✓ Reasoning parser enabled (qwen3)"
fi

# Start vLLM server
echo "Executing: $VLLM_CMD"
echo ""

eval "$VLLM_CMD 2>&1 | tee \"$LOG_FILE\""

# Note: This script will block. Press Ctrl+C to stop the server.
