#!/usr/bin/env python3
"""
Quick test to verify data loading works with both formats
"""

import sys
import os
sys.path.insert(0, 'src')

from strategy_kg import StrategyKnowledgeGraph

def test_data_loading():
    print("="*60)
    print("Testing Data Loading")
    print("="*60)

    kg = StrategyKnowledgeGraph()

    print("\nLoading data from strategy_dataset_full.json...")
    print("(This should auto-detect new format with string strategies)")

    try:
        kg.load_data_from_final_dataset(
            dataset_path='data/strategy_dataset_full.json',
            human_path=None,
            model_path=None
        )

        print("\n✅ Data loading successful!")
        print(f"\nResults:")
        print(f"  Problems loaded:   {len(kg.problems):,}")
        print(f"  Strategies loaded: {len(kg.strategies):,}")
        print(f"  Templates found:   {len(kg.templates)}")

        # Check edges
        v1_edges = len(kg.problem_strategy_edges_v1)
        v13_edges = len(kg.problem_strategy_edges_v13)
        print(f"\n  Human edges:  {v1_edges:,}")
        print(f"  Model edges:  {v13_edges:,}")
        print(f"  Total edges:  {v1_edges + v13_edges:,}")

        # Verify templates
        print(f"\n  Template categories:")
        categories = set()
        for t in kg.templates:
            categories.add(t['category'])
        for cat in sorted(categories):
            count = sum(1 for t in kg.templates if t['category'] == cat)
            print(f"    - {cat}: {count} templates")

        print("\n" + "="*60)
        print("✅ All checks passed!")
        print("="*60)

        # Show sample strategy
        if kg.strategies:
            print(f"\nSample strategy (first):")
            sample_s = kg.strategies[0]
            print(f"  Text: {sample_s[:100]}...")
            if sample_s in kg.strategy_to_template:
                info = kg.strategy_to_template[sample_s]
                print(f"  Template: {info['template']}")
                print(f"  Category: {info['category']}")
                print(f"  Source: {info['source']}")

    except Exception as e:
        print(f"\n❌ Error during data loading:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)
