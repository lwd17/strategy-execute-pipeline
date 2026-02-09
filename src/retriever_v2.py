"""
Optimized Two-Path Strategy Retrieval Pipeline

新的检索流程：
- 路径A: 新问题 semantic → 找5个相似问题 → 加权得到structural → 找top-2 template → 策略
- 路径B: 新问题 semantic → 直接匹配策略（补充）
- 统一打分：结合策略、template、问题的 semantic + structural 相似度
"""

import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from collections import defaultdict


class OptimizedStrategyRetriever:
    """优化的两路检索器"""

    def __init__(self, kg_path='strategy_kg.pkl',
                 strategy_semantic_path='strategy_semantic_emb.npy',
                 strategy_structural_path='strategy_structural_emb.npy',
                 problem_semantic_path='problem_semantic_emb.npy',
                 problem_structural_path='problem_structural_emb.npy',
                 template_semantic_path='template_semantic_emb.npy',
                 template_structural_path='template_structural_emb.npy'):

        print("Loading Optimized Strategy Retriever...")

        # 加载KG
        with open(kg_path, 'rb') as f:
            self.kg = pickle.load(f)

        # 加载embeddings
        self.strategy_semantic_emb = np.load(strategy_semantic_path)
        self.strategy_structural_emb = np.load(strategy_structural_path)
        self.problem_semantic_emb = np.load(problem_semantic_path)
        self.problem_structural_emb = np.load(problem_structural_path)
        self.template_semantic_emb = np.load(template_semantic_path)
        self.template_structural_emb = np.load(template_structural_path)

        # 预归一化所有embeddings
        print("  Pre-normalizing embeddings...")
        self.strategy_semantic_emb = self._l2_normalize(self.strategy_semantic_emb)
        self.strategy_structural_emb = self._l2_normalize(self.strategy_structural_emb)
        self.problem_semantic_emb = self._l2_normalize(self.problem_semantic_emb)
        self.problem_structural_emb = self._l2_normalize(self.problem_structural_emb)
        self.template_semantic_emb = self._l2_normalize(self.template_semantic_emb)
        self.template_structural_emb = self._l2_normalize(self.template_structural_emb)

        # 构建策略元数据
        self._build_strategy_metadata()

        # 预建问题-策略索引（用于打分时查找策略所属问题）
        self._build_strategy_to_problems_index()

        # 初始化编码器
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        print(f"✓ Loaded {len(self.kg.strategies)} strategies")
        print(f"✓ Loaded {len(self.kg.templates)} templates")
        print(f"✓ Loaded {len(self.kg.problems)} problems")

    def _l2_normalize(self, embeddings):
        """L2 normalize embeddings (rows)"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def _build_strategy_metadata(self):
        """构建策略元数据"""
        print("  Building strategy metadata...")
        self.strategy_metadata = []

        for strat_idx, strat_text in enumerate(self.kg.strategies):
            info = self.kg.strategy_to_template.get(strat_text)

            if info:
                template_name = info['template']
                category = info['category']
                source = info.get('source', 'unknown')
                template_idx = self.kg.template_to_idx.get(template_name, 0)
            else:
                template = self.kg.templates[0]
                template_name = template['name']
                category = template['category']
                template_idx = 0
                source = 'unknown'

            self.strategy_metadata.append({
                'strategy_idx': strat_idx,
                'strategy': strat_text,
                'template': template_name,
                'template_idx': template_idx,
                'category': category,
                'source': source
            })

        print(f"  ✓ Built metadata for {len(self.strategy_metadata)} strategies")

    def _build_strategy_to_problems_index(self):
        """预建策略→问题和问题→策略双向索引"""
        print("  Building strategy-to-problems index...")
        self.strategy_to_problems = defaultdict(list)
        self.problem_to_strategies = defaultdict(list)

        for edge in self.kg.problem_strategy_edges:
            prob_idx, strat_idx, is_correct = edge[0], edge[1], edge[2]
            self.strategy_to_problems[strat_idx].append((prob_idx, is_correct))
            if is_correct:  # 只索引正确的策略
                self.problem_to_strategies[prob_idx].append(strat_idx)

        print(f"  ✓ Indexed {len(self.strategy_to_problems)} strategies, {len(self.problem_to_strategies)} problems")

    def _decide_source(self, subject: str, candidate_template_indices: List[int]) -> str:
        """
        根据KG中的source_preference决定使用human还是model策略

        Args:
            subject: 问题学科 (normalized to lowercase)
            candidate_template_indices: 候选模板索引列表

        Returns:
            'human', 'model', or 'both'
        """
        if not hasattr(self.kg, 'strategy_source_preference'):
            return 'both'  # 如果KG没有preference数据，使用全部

        # 统计candidate templates中各source的preference
        source_votes = defaultdict(int)

        for template_idx in candidate_template_indices:
            template_name = self.kg.templates[template_idx]['name']
            key = (subject, template_name)

            preferred = self.kg.strategy_source_preference.get(key, 'both')
            if preferred in ['human', 'model']:
                source_votes[preferred] += 1
            else:
                source_votes['human'] += 0.5
                source_votes['model'] += 0.5

        # 如果没有明确偏好，返回both
        if not source_votes:
            return 'both'

        # 如果某个source占优(>60%)，返回该source
        total = sum(source_votes.values())
        if source_votes.get('human', 0) / total > 0.6:
            return 'human'
        elif source_votes.get('model', 0) / total > 0.6:
            return 'model'
        else:
            return 'both'

    def retrieve(self, problem_text: str, subject: str = None, k=5,
                 top_templates=2, top_similar_problems=5,
                 semantic_weight=0.5, structural_weight=0.5,
                 diversity_control=True, ann_top=50,
                 preferred_source='auto') -> List[Dict]:
        """
        优化的两路检索策略

        Args:
            problem_text: 问题文本
            k: 最终返回策略数量
            top_templates: template级召回数量（路径A使用）
            top_similar_problems: 相似问题数量（路径A使用）
            semantic_weight: 语义相似度权重
            structural_weight: 结构相似度权重
            diversity_control: 是否启用多样性控制
            ann_top: Semantic ANN召回数量（路径B使用）

        Returns:
            List of strategy dicts with scores
        """

        # Step 1: 编码新问题的 semantic embedding
        problem_semantic = self.encoder.encode([problem_text])[0]
        problem_semantic_norm = problem_semantic / (np.linalg.norm(problem_semantic) + 1e-8)

        # Step 2: 找 top-N 相似问题（用 semantic）- 三条路径都需要
        sims = np.dot(self.problem_semantic_emb, problem_semantic_norm)
        top_k_indices = np.argsort(sims)[-top_similar_problems:]
        weights = sims[top_k_indices]

        # Step 3: 路径A - 相似问题的策略
        print(f"  Path A: Strategies from {top_similar_problems} similar problems...")
        similar_problem_candidates_A = set()
        if weights.max() > 0.1:  # 至少有一个相似的
            for prob_idx in top_k_indices:
                # 使用预建的索引：O(1) 查找
                if prob_idx in self.problem_to_strategies:
                    similar_problem_candidates_A.update(self.problem_to_strategies[prob_idx])
            print(f"    Found {len(similar_problem_candidates_A)} strategies from similar problems")
        else:
            print(f"    Path A failed: no similar problems found")

        # Step 4: 路径B - 通过相似问题 → structural → template → 策略
        print(f"  Path B: Similar problems → structural → templates...")
        problem_structural_norm = None
        template_candidates_B = set()
        top_template_indices = []

        if weights.max() > 0.1:  # 至少有一个相似的
            # 4.1: 用这些问题的 structural embedding 加权平均得到新问题的 structural
            weights_exp = np.exp(weights - weights.max())
            weights_norm = weights_exp / weights_exp.sum()

            problem_structural = np.average(
                self.problem_structural_emb[top_k_indices],
                axis=0,
                weights=weights_norm
            )
            problem_structural_norm = problem_structural / (np.linalg.norm(problem_structural) + 1e-8)

            # 4.2: 用 structural 找 top-K template
            template_sims = np.dot(self.template_structural_emb, problem_structural_norm)
            top_template_indices = np.argsort(template_sims)[-top_templates:].tolist()

            # 4.3: 从这些 template 中的策略，再用 semantic 匹配取 top-20
            # 先不做source过滤，收集所有候选
            for template_idx in top_template_indices:
                # 收集该 template 的所有策略索引
                template_strategy_indices = []
                for meta in self.strategy_metadata:
                    if meta['template_idx'] == template_idx:
                        template_strategy_indices.append(meta['strategy_idx'])

                # 用问题的 semantic 对这些策略打分
                if template_strategy_indices:
                    strategy_indices_array = np.array(template_strategy_indices)
                    strategy_sims = np.dot(
                        self.strategy_semantic_emb[strategy_indices_array],
                        problem_semantic_norm
                    )
                    # 取 top-20
                    top_20_indices = np.argsort(strategy_sims)[-20:]
                    template_candidates_B.update(strategy_indices_array[top_20_indices].tolist())

            print(f"    Found {len(template_candidates_B)} strategies from {top_templates} templates (top-20 per template, before source filter)")
        else:
            print(f"    Path B failed: no similar problems found")

        # Step 4.5: Decide source preference per template (after Path B)
        # 为每个template单独决定source preference
        template_source_map = {}  # {template_idx: 'human'/'model'/'both'}

        if preferred_source == 'auto':
            if not hasattr(self.kg, 'strategy_source_preference') or not self.kg.strategy_source_preference:
                # 没有preference数据，全部用both
                for template_idx in top_template_indices:
                    template_source_map[template_idx] = 'both'
                print(f"  No source preference data, using 'both' for all templates")
            else:
                # 为每个template查询preference
                normalized_subject = subject.lower() if subject else 'unknown'
                for template_idx in top_template_indices:
                    template_name = self.kg.templates[template_idx]['name']
                    key = (normalized_subject, template_name)
                    preferred = self.kg.strategy_source_preference.get(key, 'both')
                    template_source_map[template_idx] = preferred
                    print(f"  Template '{template_name}': prefer {preferred}")
        elif preferred_source in ['human', 'model', 'both']:
            # 用户指定source，所有template统一使用
            for template_idx in top_template_indices:
                template_source_map[template_idx] = preferred_source
            print(f"  User-specified source: {preferred_source} for all templates")
        else:
            # Invalid preferred_source，fallback to 'both'
            for template_idx in top_template_indices:
                template_source_map[template_idx] = 'both'
            print(f"  Invalid preferred_source '{preferred_source}', using 'both'")

        # Step 5: 路径C - 直接用 semantic 匹配策略（补充）
        print(f"  Path C: Direct semantic matching...")
        strategy_sem_sims = np.dot(self.strategy_semantic_emb, problem_semantic_norm)
        top_semantic_indices = np.argsort(strategy_sem_sims)[-ann_top:]
        semantic_candidates_C = set(top_semantic_indices.tolist())
        print(f"    Found {len(semantic_candidates_C)} strategies from semantic matching")

        # Step 6: 合并三路候选
        all_candidates = list(similar_problem_candidates_A | template_candidates_B | semantic_candidates_C)

        if not all_candidates:
            # Fallback
            all_candidates = list(range(min(100, len(self.kg.strategies))))

        print(f"  Total candidates before source filter: {len(all_candidates)}")

        # Step 6.5: Template-level source filtering
        # 根据每个策略的template，应用对应的source preference
        if template_source_map:
            filtered_candidates = []
            filter_stats = {'kept': 0, 'filtered': 0, 'no_template': 0}

            for idx in all_candidates:
                strategy_meta = self.strategy_metadata[idx]
                strategy_source = strategy_meta['source']
                strategy_template_idx = strategy_meta['template_idx']

                # 获取该template的source preference
                preferred = template_source_map.get(strategy_template_idx, 'both')

                # 判断是否保留该策略
                if preferred == 'both':
                    # 该template允许所有source
                    filtered_candidates.append(idx)
                    filter_stats['kept'] += 1
                elif preferred == strategy_source:
                    # 策略的source匹配template的preference
                    filtered_candidates.append(idx)
                    filter_stats['kept'] += 1
                else:
                    # 策略的source不匹配，过滤掉
                    filter_stats['filtered'] += 1

            all_candidates = filtered_candidates
            print(f"  After template-level source filter: {len(all_candidates)} candidates (kept: {filter_stats['kept']}, filtered: {filter_stats['filtered']})")
        else:
            print(f"  No template-level source filter: {len(all_candidates)} candidates")

        # 记录候选来源（优先级：A > B > C）
        candidate_source = {}
        for idx in semantic_candidates_C:
            if idx in all_candidates:  # 只记录过滤后的候选
                candidate_source[idx] = 'semantic'
        for idx in template_candidates_B:
            if idx in all_candidates:
                candidate_source[idx] = 'template'  # 覆盖
        for idx in similar_problem_candidates_A:
            if idx in all_candidates:
                candidate_source[idx] = 'similar_problem'  # 最高优先级

        # Step 7: 统一打分 - 结合策略、template、问题的 embedding 相似度
        scores = self._score_strategies_unified(
            all_candidates,
            problem_semantic_norm,
            problem_structural_norm,
            semantic_weight,
            structural_weight,
            candidate_source,
            template_source_map  # 传递template-level的source map
        )

        # Step 6: 去重和多样性控制
        if diversity_control:
            final_results = self._apply_diversity_control(scores, k)
        else:
            top_indices = np.argsort(scores)[-k:][::-1]
            final_results = [
                {
                    **self.strategy_metadata[idx],
                    'score': float(scores[idx])
                }
                for idx in top_indices
            ]

        return final_results[:k]

    def _score_strategies_unified(self, candidate_indices, problem_semantic_norm,
                                  problem_structural_norm, sem_weight, struct_weight,
                                  candidate_source, template_source_map=None):
        """
        统一打分：结合策略、template、问题的 embedding 相似度

        打分公式：
        score = w_sem * (strategy_sem_sim + template_sem_sim + avg_problem_sem_sim) / 3
              + w_struct * (strategy_struct_sim + template_struct_sim + avg_problem_struct_sim) / 3
              + recall_boost
              + source_bias (template-level)

        Args:
            template_source_map: {template_idx: 'human'/'model'/'both'} - template级别的source preference
        """
        scores = np.zeros(len(self.kg.strategies), dtype=np.float32)

        for idx in candidate_indices:
            meta = self.strategy_metadata[idx]
            template_idx = meta['template_idx']

            # 1. 策略 semantic 相似度
            strategy_sem_sim = np.dot(self.strategy_semantic_emb[idx], problem_semantic_norm)

            # 2. Template semantic 相似度
            template_sem_sim = np.dot(self.template_semantic_emb[template_idx], problem_semantic_norm)

            # 3. 策略所属问题的平均 semantic 相似度
            avg_problem_sem_sim = 0.0
            if idx in self.strategy_to_problems:
                problem_indices = [p_idx for p_idx, _ in self.strategy_to_problems[idx]]
                if problem_indices:
                    problem_sems = self.problem_semantic_emb[problem_indices]
                    sims = np.dot(problem_sems, problem_semantic_norm)
                    avg_problem_sem_sim = sims.mean()

            # 4. Semantic 总分
            semantic_score = (strategy_sem_sim + template_sem_sim + avg_problem_sem_sim) / 3.0

            # 5. Structural 分数（如果可用）
            structural_score = 0.0
            if problem_structural_norm is not None:
                # 策略 structural 相似度
                strategy_struct_sim = np.dot(self.strategy_structural_emb[idx], problem_structural_norm)

                # Template structural 相似度
                template_struct_sim = np.dot(self.template_structural_emb[template_idx], problem_structural_norm)

                # 策略所属问题的平均 structural 相似度
                avg_problem_struct_sim = 0.0
                if idx in self.strategy_to_problems:
                    problem_indices = [p_idx for p_idx, _ in self.strategy_to_problems[idx]]
                    if problem_indices:
                        problem_structs = self.problem_structural_emb[problem_indices]
                        sims = np.dot(problem_structs, problem_structural_norm)
                        avg_problem_struct_sim = sims.mean()

                structural_score = (strategy_struct_sim + template_struct_sim + avg_problem_struct_sim) / 3.0

            # 6. 加权合并
            if problem_structural_norm is not None:
                base_score = sem_weight * semantic_score + struct_weight * structural_score
            else:
                base_score = semantic_score  # 只用 semantic

            # 7. Recall boost（根据召回路径，优先级：similar_problem > template > semantic）
            source = candidate_source.get(idx, 'semantic')
            if source == 'similar_problem':
                recall_boost = 0.15  # 最高优先级
            elif source == 'template':
                recall_boost = 0.10  # 中等优先级
            else:  # semantic
                recall_boost = 0.05  # 最低优先级

            # 8. Template-level source preference bias
            # 根据该策略所属template的preference，应用source bias
            source_bias = 0.0
            if template_source_map:
                preferred = template_source_map.get(template_idx, 'both')
                strategy_source = meta['source']

                if preferred in ['human', 'model']:
                    if strategy_source == preferred:
                        source_bias = 0.15  # Strong boost for preferred source
                    elif strategy_source != 'unknown':
                        source_bias = -0.08  # Penalty for non-preferred source
                # 如果preferred='both'，不应用bias

            scores[idx] = base_score + recall_boost + source_bias

        return scores

    def _apply_diversity_control(self, scores, k):
        """多样性控制"""
        sorted_indices = np.argsort(scores)[::-1]

        selected = []
        template_counts = defaultdict(int)
        category_counts = defaultdict(int)
        seen_texts = set()

        min_score_threshold = 0.5
        max_per_template = 1 if k <= 5 else 2
        max_per_category = 2 if k <= 5 else 3

        for idx in sorted_indices:
            if len(selected) >= k:
                break

            score = scores[idx]
            if score < min_score_threshold:
                continue

            meta = self.strategy_metadata[idx]
            template = meta['template']
            category = meta['category']
            strat_text = meta['strategy']

            if strat_text in seen_texts:
                continue

            if template_counts[template] >= max_per_template:
                continue
            if category_counts[category] >= max_per_category:
                continue

            selected.append({
                **meta,
                'score': float(scores[idx])
            })
            template_counts[template] += 1
            category_counts[category] += 1
            seen_texts.add(strat_text)

        return selected[:k]