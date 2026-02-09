"""
Strategy Retriever using Train Set Knowledge Graph

This retriever uses the KG built from train set only (4832 problems),
ensuring no data leakage when testing on the 100-problem test set.
"""

import os
from retriever_v2 import OptimizedStrategyRetriever


class TrainSetRetriever(OptimizedStrategyRetriever):
    """Retriever using train set KG only"""

    def __init__(self):
        """Initialize retriever with train set artifacts"""
        print("Loading Train Set Strategy Retriever...")

        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize with train set files
        super().__init__(
            kg_path=os.path.join(current_dir, 'strategy_kg_train.pkl'),
            strategy_semantic_path=os.path.join(current_dir, 'strategy_semantic_emb_train.npy'),
            strategy_structural_path=os.path.join(current_dir, 'strategy_structural_emb_train.npy'),
            problem_semantic_path=os.path.join(current_dir, 'problem_semantic_emb_train.npy'),
            problem_structural_path=os.path.join(current_dir, 'problem_structural_emb_train.npy'),
            template_semantic_path=os.path.join(current_dir, 'template_semantic_emb_train.npy'),
            template_structural_path=os.path.join(current_dir, 'template_structural_emb_train.npy')
        )

        print("✓ Train Set Retriever ready (no test set data leakage)")