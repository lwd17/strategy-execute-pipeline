#!/usr/bin/env python3
"""
Build Knowledge Graph from Train Set Only

Usage:
    python build_kg_train_only.py

This script builds a KG using only the 4832 training problems,
excluding the 100 test problems to avoid data leakage.
"""

import torch
import numpy as np
import pickle
from strategy_kg import (
    StrategyKnowledgeGraph,
    SemanticEmbedder,
    ContrastiveGNN,
    StrategyRetriever,
    StrategySuitabilityClassifier,
    contrastive_loss
)
from collections import defaultdict


def train_gnn(gnn_model, kg, semantic_embedder, device='cpu',
              num_epochs=50, batch_size=32, lr=0.001):
    """Train GNN with contrastive learning"""
    print(f"\n{'='*70}")
    print("Training GNN with Contrastive Learning")
    print(f"{'='*70}\n")

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)

    # Prepare data
    problem_emb = semantic_embedder.encode_problems(kg.problems)
    strategy_emb = semantic_embedder.encode_strategies(kg.strategies)

    # ✅ FIX: Use descriptive text for template embedding, not just template names
    template_texts = [
        f"{t['name'].replace('_', ' ')}. {t['category']}. {t['description']}"
        for t in kg.templates
    ]
    template_emb = semantic_embedder.model.encode(template_texts)

    x_dict = {
        'problem': torch.FloatTensor(problem_emb).to(device),
        'strategy': torch.FloatTensor(strategy_emb).to(device),
        'template': torch.FloatTensor(template_emb).to(device),
    }

    # Build edges
    ps_edges = []
    ps_attrs = []
    # ✅ FIX 6: 保留wrong edges，但使用不同权重
    # correct: weight=1.0 (强连接)
    # wrong: weight=0.3 (弱连接，也提供"这个组合不work"的信号)
    # 不要丢掉wrong edges！它们也是valuable signal
    for edge in kg.problem_strategy_edges:
        p_idx, s_idx, is_correct = edge[0], edge[1], edge[2]
        ps_edges.append([p_idx, s_idx])
        # ✅ FIX 6: 使用weighted edges
        edge_weight = 1.0 if is_correct else 0.3
        ps_attrs.append([edge_weight])  # correct=1.0, wrong=0.3

    st_edges = []
    for s_idx, t_idx in kg.strategy_template_edges:
        st_edges.append([s_idx, t_idx])

    edge_index_dict = {
        ('problem', 'uses', 'strategy'): torch.LongTensor(ps_edges).t().to(device),
        ('strategy', 'belongs_to', 'template'): torch.LongTensor(st_edges).t().to(device),
    }

    edge_attr_dict = {
        ('problem', 'uses', 'strategy'): torch.FloatTensor(ps_attrs).to(device),
    }

    # Build training pairs
    problem_strategies = defaultdict(list)
    # ✅ FIX: edges now have 4 elements (p_idx, s_idx, is_correct, source)
    for edge in kg.problem_strategy_edges:
        p_idx, s_idx, is_correct = edge[0], edge[1], edge[2]
        if is_correct:  # Only use correct strategies for positive pairs
            problem_strategies[p_idx].append(s_idx)

    # ✅ FIX D: 构建所有策略列表，用于负采样（允许1个正样本的题也能训练）
    all_strategy_indices = list(range(len(kg.strategies)))

    # Training loop
    gnn_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Sample mini-batches
        problem_ids = list(problem_strategies.keys())
        np.random.shuffle(problem_ids)

        for i in range(0, len(problem_ids), batch_size):
            batch_problems = problem_ids[i:i+batch_size]

            # Forward pass
            h_dict = gnn_model(x_dict, edge_index_dict, edge_attr_dict)

            batch_loss = None  # 初始化为None而不是0
            for p_idx in batch_problems:
                # ✅ FIX D: 允许只有1个正样本的题（从全局/其他题采负样本）
                if len(problem_strategies[p_idx]) < 1:
                    continue

                # Get problem embedding
                p_emb = h_dict['problem'][p_idx]

                # Sample positive from this problem's correct strategies
                pos_s_idx = np.random.choice(problem_strategies[p_idx])
                pos_emb = h_dict['strategy'][pos_s_idx]

                # ✅ FIX D: 改进负采样 - 从其他问题的策略或全局策略中采样
                # 优先从其他问题采，如果失败则从全局采（排除当前问题的正样本）
                neg_s_idx = None
                # 尝试从其他问题采样
                other_problems = [pid for pid in problem_ids if pid != p_idx]
                if other_problems:
                    neg_p_idx = np.random.choice(other_problems)
                    if len(problem_strategies[neg_p_idx]) > 0:
                        neg_s_idx = np.random.choice(problem_strategies[neg_p_idx])

                # 如果失败，从全局策略采样（排除当前问题的正样本）
                if neg_s_idx is None:
                    excluded = set(problem_strategies[p_idx])
                    candidates = [s for s in all_strategy_indices if s not in excluded]
                    if len(candidates) == 0:
                        continue
                    neg_s_idx = np.random.choice(candidates)

                neg_emb = h_dict['strategy'][neg_s_idx]

                # Compute loss
                loss = contrastive_loss(
                    p_emb.unsqueeze(0),
                    pos_emb.unsqueeze(0),
                    neg_emb.unsqueeze(0)
                )

                # 累加loss，确保batch_loss始终是tensor
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss = batch_loss + loss

            # 使用item()转换为Python数值再判断
            if batch_loss is not None and batch_loss.item() > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print(f"\n✓ GNN training completed!")
    return gnn_model


def compute_embeddings(kg, semantic_embedder, gnn_model, device='cpu'):
    """Compute all embeddings"""
    print("\nComputing embeddings...")

    # Semantic embeddings
    problem_semantic = semantic_embedder.encode_problems(kg.problems)
    strategy_semantic = semantic_embedder.encode_strategies(kg.strategies)

    # ✅ FIX: Use descriptive text for template embedding
    template_texts = [
        f"{t['name'].replace('_', ' ')}. {t['category']}. {t['description']}"
        for t in kg.templates
    ]
    template_semantic = semantic_embedder.model.encode(template_texts)

    print(f"✓ Computed semantic embeddings")

    # Structural embeddings via GNN
    gnn_model.eval()

    x_dict = {
        'problem': torch.FloatTensor(problem_semantic).to(device),
        'strategy': torch.FloatTensor(strategy_semantic).to(device),
        'template': torch.FloatTensor(template_semantic).to(device),
    }

    # Build edges (for compute_embeddings, same logic as train_gnn)
    ps_edges = []
    ps_attrs = []
    # ✅ FIX 6: 保留wrong edges，使用权重
    for edge in kg.problem_strategy_edges:
        p_idx, s_idx, is_correct = edge[0], edge[1], edge[2]
        ps_edges.append([p_idx, s_idx])
        edge_weight = 1.0 if is_correct else 0.3
        ps_attrs.append([edge_weight])

    st_edges = []
    for s_idx, t_idx in kg.strategy_template_edges:
        st_edges.append([s_idx, t_idx])

    edge_index_dict = {
        ('problem', 'uses', 'strategy'): torch.LongTensor(ps_edges).t().to(device),
        ('strategy', 'belongs_to', 'template'): torch.LongTensor(st_edges).t().to(device),
    }

    edge_attr_dict = {
        ('problem', 'uses', 'strategy'): torch.FloatTensor(ps_attrs).to(device),
    }

    with torch.no_grad():
        h_dict = gnn_model(x_dict, edge_index_dict, edge_attr_dict)

    problem_structural = h_dict['problem'].cpu().numpy()
    strategy_structural = h_dict['strategy'].cpu().numpy()
    template_structural = h_dict['template'].cpu().numpy()

    print(f"✓ Computed structural embeddings via GNN")

    # ✅ FIX: L2 normalize all embeddings before saving
    def l2norm(x):
        """L2 normalize rows of matrix"""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-8)

    print(f"  Normalizing all embeddings...")
    problem_semantic = l2norm(problem_semantic)
    strategy_semantic = l2norm(strategy_semantic)
    template_semantic = l2norm(template_semantic)
    problem_structural = l2norm(problem_structural)
    strategy_structural = l2norm(strategy_structural)
    template_structural = l2norm(template_structural)

    print(f"✓ All embeddings L2-normalized")

    return {
        'problem_semantic': problem_semantic,
        'strategy_semantic': strategy_semantic,
        'template_semantic': template_semantic,
        'problem_structural': problem_structural,
        'strategy_structural': strategy_structural,
        'template_structural': template_structural,
    }


def main():
    print("="*70)
    print("BUILDING STRATEGY KNOWLEDGE GRAPH (TRAIN SET ONLY)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # 1. Build knowledge graph
    print("\n" + "="*70)
    print("Step 1: Building Knowledge Graph from Train Set")
    print("="*70)

    kg = StrategyKnowledgeGraph()
    kg.load_data_from_final_dataset(
        dataset_path='./strategy_dataset_train.json',  # ✅ Use train set only
        human_path='preference_human.json',  # Path to human strategy preference file
        model_path='preference_model.json'   # Path to model strategy preference file
    )

    print(f"\n✓ Loaded knowledge graph from train set")
    print(f"  - Total problems: {len(kg.problems)}")
    print(f"  - Total strategies: {len(kg.strategies)}")
    print(f"  - Total templates: {len(kg.templates)}")

    # 2. Initialize semantic embedder
    print("\n" + "="*70)
    print("Step 2: Initializing Semantic Embedder")
    print("="*70)

    semantic_embedder = SemanticEmbedder(device=device)

    # 3. Initialize and train GNN
    print("\n" + "="*70)
    print("Step 3: Training GNN")
    print("="*70)

    gnn_model = ContrastiveGNN(
        input_dim=384,  # SentenceTransformer dimension
        hidden_dim=128,
        output_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    ).to(device)

    gnn_model = train_gnn(
        gnn_model, kg, semantic_embedder,
        device=device,
        num_epochs=50,
        batch_size=32,
        lr=0.001
    )

    # 4. Compute embeddings
    print("\n" + "="*70)
    print("Step 4: Computing Embeddings")
    print("="*70)

    embeddings = compute_embeddings(kg, semantic_embedder, gnn_model, device)

    # 5. Train classifier
    print("\n" + "="*70)
    print("Step 5: Training Classifier")
    print("="*70)

    classifier = StrategySuitabilityClassifier()
    classifier.train(
        kg,
        embeddings['problem_semantic'],
        embeddings['strategy_semantic'],
        embeddings['problem_structural'],
        embeddings['strategy_structural'],
        embeddings['template_structural'],
        embeddings['template_semantic']
    )

    # 6. Save everything
    print("\n" + "="*70)
    print("Step 6: Saving Results")
    print("="*70)

    # Save knowledge graph
    with open('strategy_kg_train.pkl', 'wb') as f:
        pickle.dump(kg, f)
    print("✓ Saved knowledge graph: strategy_kg_train.pkl")

    # Save GNN model
    torch.save(gnn_model.state_dict(), 'gnn_model_train.pth')
    print("✓ Saved GNN model: gnn_model_train.pth")

    # Save embeddings
    for name, emb in embeddings.items():
        np.save(f'{name}_emb_train.npy', emb)
        print(f"✓ Saved {name}_emb_train.npy")

    # Save classifier
    with open('strategy_classifier_train.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("✓ Saved classifier: strategy_classifier_train.pkl")

    print("\n" + "="*70)
    print("✅ KNOWLEDGE GRAPH BUILD COMPLETE (TRAIN SET ONLY)!")
    print("="*70)
    print("\nGenerated files:")
    print("  - strategy_kg_train.pkl (knowledge graph)")
    print("  - gnn_model_train.pth (trained GNN)")
    print("  - *_emb_train.npy (6 embedding files)")
    print("  - strategy_classifier_train.pkl (classifier)")
    print("\nThese files contain NO information from the 100-problem test set.")


if __name__ == '__main__':
    main()