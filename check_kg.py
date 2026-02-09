#!/usr/bin/env python3
"""
Quick script to check which knowledge graph is loaded
"""

import pickle
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_kg():
    kg_path = 'strategy_kg.pkl'

    if not os.path.exists(kg_path):
        print("❌ No knowledge graph found!")
        print("   Run 'python src/build_kg.py' to create one")
        return

    print("Loading knowledge graph...")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)

    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    print(f"Problems:   {len(kg.problems):,}")
    print(f"Strategies: {len(kg.strategies):,}")
    print(f"Templates:  {len(kg.templates)}")

    # Analyze edge counts
    v1_edges = len(kg.problem_strategy_edges_v1) if hasattr(kg, 'problem_strategy_edges_v1') else 0
    v13_edges = len(kg.problem_strategy_edges_v13) if hasattr(kg, 'problem_strategy_edges_v13') else 0

    print(f"\nEdges:")
    print(f"  Human strategies:  {v1_edges:,}")
    print(f"  Model strategies:  {v13_edges:,}")
    print(f"  Total:             {v1_edges + v13_edges:,}")

    # Determine which version
    print("\n" + "="*60)
    if len(kg.problems) >= 4900:
        print("✅ Using HYBRID_KG_SYSTEM pre-trained models")
        print("   Baseline: 73.33% on AIME 2025 (22/30)")
    elif len(kg.problems) >= 4800:
        print("✅ Using newly trained KG from full dataset")
    elif len(kg.problems) >= 100:
        print("✅ Using newly trained KG from train split")
    else:
        print("⚠️  Small KG - may have limited coverage")
    print("="*60)

if __name__ == '__main__':
    check_kg()
