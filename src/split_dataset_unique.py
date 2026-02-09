#!/usr/bin/env python3
"""
Split strategy dataset into train and test sets (based on unique problem texts)
- Randomly sample 100 unique problem texts for testing
- Remaining unique texts for building knowledge graph
- Handle duplicate problem texts properly (all variants go to same split)
- Save both splits for reproducibility
"""

import json
import random
import os

def split_dataset(
    input_path: str = '../data/strategy_dataset_full.json',
    train_output_path: str = '../strategy_dataset_train.json',
    test_output_path: str = '../strategy_dataset_test.json',
    test_size: int = 100,
    random_seed: int = 42
):
    """Split dataset into train and test sets (based on unique problem texts)"""

    print("="*70)
    print("SPLITTING DATASET INTO TRAIN/TEST (UNIQUE TEXTS)")
    print("="*70)

    # Load original dataset
    print(f"\n[1/5] Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data['metadata']
    all_problems = data['data']

    print(f"✓ Loaded {len(all_problems)} problems")
    print(f"  - Total human strategies: {metadata['total_human_strategies']}")
    print(f"  - Total model strategies: {metadata['total_model_strategies']}")

    # Group problems by unique text (to handle duplicates)
    print(f"\n[2/5] Grouping problems by unique text...")
    text_to_problems = {}
    for prob in all_problems:
        text = prob['problem_text']
        if text not in text_to_problems:
            text_to_problems[text] = []
        text_to_problems[text].append(prob)

    unique_texts = list(text_to_problems.keys())
    print(f"✓ Found {len(unique_texts)} unique problem texts")
    print(f"  - {len(all_problems) - len(unique_texts)} duplicate texts in original dataset")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Randomly sample unique texts for test set
    print(f"\n[3/5] Randomly sampling {test_size} unique texts for test set (seed={random_seed})...")
    random.shuffle(unique_texts)

    test_texts = set(unique_texts[:test_size])
    train_texts = set(unique_texts[test_size:])

    print(f"✓ Selected {len(test_texts)} unique texts for test")
    print(f"✓ Remaining {len(train_texts)} unique texts for train")

    # Split problems (all variants of same text go to same split)
    print(f"\n[4/5] Splitting problems...")
    test_problems = []
    train_problems = []

    test_human_strategies = 0
    test_model_strategies = 0
    train_human_strategies = 0
    train_model_strategies = 0

    for text in test_texts:
        for prob in text_to_problems[text]:
            test_problems.append(prob)
            test_human_strategies += len(prob.get('human_strategies', []))
            test_model_strategies += len(prob.get('model_strategies', []))

    for text in train_texts:
        for prob in text_to_problems[text]:
            train_problems.append(prob)
            train_human_strategies += len(prob.get('human_strategies', []))
            train_model_strategies += len(prob.get('model_strategies', []))

    print(f"✓ Test set: {len(test_problems)} problems ({len(test_texts)} unique texts)")
    print(f"  - Human strategies: {test_human_strategies}")
    print(f"  - Model strategies: {test_model_strategies}")
    print(f"✓ Train set: {len(train_problems)} problems ({len(train_texts)} unique texts)")
    print(f"  - Human strategies: {train_human_strategies}")
    print(f"  - Model strategies: {train_model_strategies}")

    # Create output data structures
    test_data = {
        "metadata": {
            "total_problems": len(test_problems),
            "unique_problem_texts": len(test_texts),
            "total_human_strategies": test_human_strategies,
            "total_model_strategies": test_model_strategies,
            "format": "compact",
            "description": "Test set: 100 randomly sampled unique problem texts with ground truth answers",
            "random_seed": random_seed,
            "split_from": input_path
        },
        "data": test_problems
    }

    train_data = {
        "metadata": {
            "total_problems": len(train_problems),
            "unique_problem_texts": len(train_texts),
            "total_human_strategies": train_human_strategies,
            "total_model_strategies": train_model_strategies,
            "format": "compact",
            "description": "Train set: remaining unique problem texts for building knowledge graph",
            "random_seed": random_seed,
            "split_from": input_path
        },
        "data": train_problems
    }

    # Save splits
    print(f"\n[5/5] Saving splits...")

    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved train set to: {train_output_path}")

    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved test set to: {test_output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    print(f"Original dataset: {len(all_problems)} problems ({len(unique_texts)} unique texts)")
    print(f"  ├─ Train set: {len(train_problems)} problems ({len(train_texts)} unique texts)")
    print(f"  └─ Test set:  {len(test_problems)} problems ({len(test_texts)} unique texts)")
    print(f"\nRandom seed: {random_seed}")
    print(f"Train set: {train_output_path}")
    print(f"Test set:  {test_output_path}")
    print("="*70)

    # Verify no overlap in texts
    train_text_set = set(p['problem_text'] for p in train_problems)
    test_text_set = set(p['problem_text'] for p in test_problems)
    overlap_texts = train_text_set & test_text_set

    if overlap_texts:
        print(f"\n⚠ WARNING: Found {len(overlap_texts)} overlapping problem texts!")
        print("This should not happen!")
    else:
        print(f"\n✓ Verified: No overlap in problem texts between train and test sets")


if __name__ == '__main__':
    split_dataset()