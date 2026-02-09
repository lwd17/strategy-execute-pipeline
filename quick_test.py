#!/usr/bin/env python3
"""
Quick test to verify the pipeline is working correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from retriever_v2 import OptimizedStrategyRetriever

def main():
    print("=" * 70)
    print("Quick System Test - Math Strategy Pipeline")
    print("=" * 70)

    # Test 1: Load retriever
    print("\n[Test 1] Loading pre-trained retriever...")
    try:
        retriever = OptimizedStrategyRetriever()
        print(f"✓ Retriever loaded successfully!")
        print(f"  - Problems: {len(retriever.kg.problems)}")
        print(f"  - Strategies: {len(retriever.kg.strategies)}")
        print(f"  - Templates: {len(retriever.kg.templates)}")
    except Exception as e:
        print(f"✗ Failed to load retriever: {e}")
        return False

    # Test 2: Retrieve strategies for a sample problem
    print("\n[Test 2] Testing retrieval on a sample problem...")
    test_problem = """
    Find the number of positive integers n ≤ 1000 such that n is divisible
    by both 2 and 3 but not divisible by 12.
    """

    try:
        strategies = retriever.retrieve(
            problem_text=test_problem,
            subject="number_theory",
            k=5
        )
        print(f"✓ Retrieved {len(strategies)} strategies")
        print("\nTop 3 strategies:")
        for i, s in enumerate(strategies[:3], 1):
            print(f"\n  {i}. [{s['template']}] (score: {s['score']:.3f})")
            print(f"     {s['strategy'][:100]}...")
            print(f"     Source: {s['source']}, Category: {s['category']}")
    except Exception as e:
        print(f"✗ Failed to retrieve strategies: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start vLLM server: bash scripts/start_vllm_server.sh")
    print("  2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("  3. Run benchmarks: python tests/test_aime25_accuracy.py")
    print()

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
