#!/bin/bash
# Build knowledge graph from toy strategy dataset

set -e

echo "============================================================"
echo "Building Knowledge Graph"
echo "============================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check if dataset exists
echo ""
echo "[1/2] Checking dataset..."
if [ ! -f "data/strategy_dataset_full.json" ]; then
    echo "✗ Dataset not found: data/strategy_dataset_full.json"
    echo ""
    echo "Please ensure the full strategy dataset is in the data/ directory."
    echo "You can also use your own dataset by editing src/build_kg.py"
    echo ""
    exit 1
fi
echo "✓ Dataset found"

# Build KG
echo ""
echo "[2/2] Building knowledge graph..."
echo "  This will take 10-20 minutes on GPU, 30-60 minutes on CPU"
echo "  Output files will be saved in current directory"
echo ""

python src/build_kg.py

echo ""
echo "============================================================"
echo "✓ Knowledge graph build completed!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - strategy_kg.pkl (knowledge graph)"
echo "  - gnn_model.pth (trained GNN)"
echo "  - problem_semantic_emb.npy (semantic embeddings)"
echo "  - problem_structural_emb.npy (structural embeddings)"
echo "  - strategy_semantic_emb.npy"
echo "  - strategy_structural_emb.npy"
echo "  - template_semantic_emb.npy"
echo "  - template_structural_emb.npy"
echo "  - strategy_classifier.pkl (classifier)"
echo ""
echo "You can now run tests:"
echo "  bash scripts/run_aime25.sh"
echo "  bash scripts/run_apex.sh"
echo "  bash scripts/run_hmreasoning.sh"
echo "============================================================"