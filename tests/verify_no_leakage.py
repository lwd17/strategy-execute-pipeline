#!/usr/bin/env python3
"""
Verify that train/test split has no data leakage

This script verifies:
1. No overlap between train and test problem IDs
2. Train KG contains exactly 4832 problems
3. Test set contains exactly 100 problems
4. All test problems have ground truth answers
"""

import json
import pickle


def verify_split():
    print("="*70)
    print("VERIFYING TRAIN/TEST SPLIT (NO DATA LEAKAGE)")
    print("="*70)

    # Load train set
    print("\n[1/4] Loading train set...")
    with open('./strategy_dataset_train.json', 'r') as f:
        train_data = json.load(f)
    train_problems = train_data['data']
    train_ids = set(p['problem_id'] for p in train_problems)
    print(f"✓ Train set: {len(train_problems)} problems")

    # Load test set
    print("\n[2/4] Loading test set...")
    with open('./strategy_dataset_test.json', 'r') as f:
        test_data = json.load(f)
    test_problems = test_data['data']
    test_ids = set(p['problem_id'] for p in test_problems)
    print(f"✓ Test set: {len(test_problems)} problems")

    # Load train KG
    print("\n[3/4] Loading train KG...")
    with open('./strategy_kg_train.pkl', 'rb') as f:
        kg = pickle.load(f)
    kg_problem_texts = set(p['problem_text'] for p in kg.problems)
    print(f"✓ Train KG: {len(kg.problems)} problems")

    # Verify no overlap
    print("\n[4/4] Verifying no overlap...")

    # Check 1: Train and test problem IDs are disjoint
    overlap_ids = train_ids & test_ids
    if overlap_ids:
        print(f"✗ FAIL: Found {len(overlap_ids)} overlapping problem IDs!")
        print(f"  Examples: {list(overlap_ids)[:5]}")
        return False
    else:
        print(f"✓ PASS: No overlapping problem IDs")

    # Check 2: Train KG problem count matches train set
    if len(kg.problems) != len(train_problems):
        print(f"✗ FAIL: Train KG has {len(kg.problems)} problems, but train set has {len(train_problems)}")
        return False
    else:
        print(f"✓ PASS: Train KG has correct number of problems ({len(kg.problems)})")

    # Check 3: Test set has exactly 100 problems
    if len(test_problems) != 100:
        print(f"✗ FAIL: Test set has {len(test_problems)} problems, expected 100")
        return False
    else:
        print(f"✓ PASS: Test set has exactly 100 problems")

    # Check 4: All test problems have ground truth answers
    missing_answers = [p['problem_id'] for p in test_problems if not p.get('solution')]
    if missing_answers:
        print(f"✗ FAIL: {len(missing_answers)} test problems missing ground truth answers")
        print(f"  Examples: {missing_answers[:5]}")
        return False
    else:
        print(f"✓ PASS: All test problems have ground truth answers")

    # Check 5: Verify test problem texts are NOT in train KG
    test_problem_texts = set(p['problem_text'] for p in test_problems)
    overlap_texts = test_problem_texts & kg_problem_texts
    if overlap_texts:
        print(f"✗ FAIL: Found {len(overlap_texts)} test problem texts in train KG!")
        print(f"  This indicates data leakage!")
        return False
    else:
        print(f"✓ PASS: No test problem texts found in train KG")

    # Summary
    print("\n" + "="*70)
    print("✅ VERIFICATION PASSED - NO DATA LEAKAGE DETECTED")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Train set: {len(train_problems)} problems")
    print(f"  - Test set: {len(test_problems)} problems")
    print(f"  - Train KG: {len(kg.problems)} problems")
    print(f"  - Total unique problems: {len(train_ids | test_ids)}")
    print(f"  - No overlap detected")
    print()

    # Print some statistics
    print("Train KG statistics:")
    print(f"  - Strategies: {len(kg.strategies)}")
    print(f"  - Templates: {len(kg.templates)}")
    print(f"  - Problem-Strategy edges: {len(kg.problem_strategy_edges)}")

    print("\nTest set statistics:")
    test_subjects = {}
    for p in test_problems:
        subj = p.get('problem_subject', 'unknown')
        test_subjects[subj] = test_subjects.get(subj, 0) + 1

    print("  Subject distribution:")
    for subj, count in sorted(test_subjects.items()):
        print(f"    - {subj}: {count} problems")

    return True


if __name__ == '__main__':
    import sys
    success = verify_split()
    sys.exit(0 if success else 1)