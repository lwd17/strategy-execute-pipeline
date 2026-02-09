#!/usr/bin/env python3
"""
Strategy Knowledge Graph System - Unified Implementation

Features:
1. Knowledge graph with problems, strategies, and templates
2. Semantic embeddings (SentenceTransformer)
3. Structural embeddings (trained GNN with contrastive learning)
4. Strategy template extraction (30 predefined templates)
5. Top-k retrieval with subject filtering
6. Strategy suitability classifier
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import re


# ============================================================================
# Part 1: Strategy Template Extraction
# ============================================================================

class StrategyTemplateExtractor:
    """Extract reusable templates from concrete strategies"""

    def __init__(self):
        self.template_patterns = self._build_template_patterns()

    def _build_template_patterns(self) -> Dict[str, Dict]:
        """Build template pattern library"""
        patterns = {}

        # Coordinate Methods
        patterns['coordinate_setup'] = {
            'keywords': ['coordinate', 'assign coordinates', 'parameterize', 'position'],
            'category': 'Coordinate',
            'description': 'Set up coordinate system to algebraize geometric problems'
        }
        patterns['vector_method'] = {
            'keywords': ['vector', 'position vector', 'dot product', 'cross product'],
            'category': 'Coordinate',
            'description': 'Use vector algebra and operations'
        }
        patterns['complex_number'] = {
            'keywords': ['complex number', 'complex plane', 'argand'],
            'category': 'Coordinate',
            'description': 'Represent geometric objects as complex numbers'
        }

        # Geometric Methods
        patterns['auxiliary_construction'] = {
            'keywords': ['auxiliary', 'construct', 'draw', 'add'],
            'category': 'Geometric',
            'description': 'Add auxiliary lines, circles, or points to reveal structure'
        }
        patterns['similarity_congruence'] = {
            'keywords': ['similar', 'congruent', 'sss', 'sas', 'asa', 'aas'],
            'category': 'Geometric',
            'description': 'Use triangle similarity or congruence'
        }
        patterns['angle_chasing'] = {
            'keywords': ['angle', 'inscribed angle', 'central angle', 'angle sum'],
            'category': 'Geometric',
            'description': 'Compute angles using geometric properties'
        }
        patterns['circle_properties'] = {
            'keywords': ['circle', 'cyclic', 'tangent', 'chord', 'power of point'],
            'category': 'Geometric',
            'description': 'Apply circle theorems and properties'
        }
        patterns['symmetry_analysis'] = {
            'keywords': ['symmetry', 'symmetric', 'cyclic relabel', 'invariant'],
            'category': 'Geometric',
            'description': 'Exploit symmetry to simplify or generalize'
        }

        # Number Theory Methods
        patterns['prime_factorization'] = {
            'keywords': ['prime', 'factor', 'prime power', 'decomposition', 'exponent'],
            'category': 'Number Theory',
            'description': 'Analyze via prime factorization'
        }
        patterns['modular_arithmetic'] = {
            'keywords': ['modulo', 'mod', 'congruence', 'remainder', 'residue'],
            'category': 'Number Theory',
            'description': 'Use modular arithmetic and congruences'
        }
        patterns['gcd_lcm'] = {
            'keywords': ['gcd', 'lcm', 'greatest common divisor', 'least common multiple', 'coprime'],
            'category': 'Number Theory',
            'description': 'Apply GCD/LCM properties'
        }
        patterns['divisibility'] = {
            'keywords': ['divisib', 'divide', 'divides', 'factor of'],
            'category': 'Number Theory',
            'description': 'Analyze divisibility properties'
        }

        # Combinatorial Methods
        patterns['counting_principle'] = {
            'keywords': ['count', 'combination', 'permutation', 'choose', 'c(n,k)'],
            'category': 'Combinatorial',
            'description': 'Use counting principles and binomial coefficients'
        }
        patterns['inclusion_exclusion'] = {
            'keywords': ['inclusion-exclusion', 'overcounting', 'subtract'],
            'category': 'Combinatorial',
            'description': 'Apply inclusion-exclusion principle'
        }
        patterns['pigeonhole'] = {
            'keywords': ['pigeonhole', 'drawer principle', 'average argument'],
            'category': 'Combinatorial',
            'description': 'Use pigeonhole principle'
        }
        patterns['bijection'] = {
            'keywords': ['bijection', 'one-to-one', 'correspondence', 'mapping'],
            'category': 'Combinatorial',
            'description': 'Establish bijection between sets'
        }
        patterns['probability_method'] = {
            'keywords': ['probability', 'expected value', 'random', 'complement'],
            'category': 'Combinatorial',
            'description': 'Use probability and expectation'
        }

        # Algebraic Methods
        patterns['algebraic_manipulation'] = {
            'keywords': ['expand', 'factor', 'simplify', 'substitute', 'rearrange'],
            'category': 'Algebraic',
            'description': 'Perform algebraic manipulations'
        }
        patterns['inequality'] = {
            'keywords': ['inequality', 'am-gm', 'cauchy-schwarz', 'jensen', 'bound'],
            'category': 'Algebraic',
            'description': 'Apply algebraic inequalities'
        }
        patterns['polynomial_analysis'] = {
            'keywords': ['polynomial', 'root', 'vieta', 'discriminant', 'degree'],
            'category': 'Algebraic',
            'description': 'Analyze polynomials and their roots'
        }
        patterns['symmetric_sum'] = {
            'keywords': ['symmetric sum', 'power sum', 'elementary symmetric'],
            'category': 'Algebraic',
            'description': 'Use symmetric functions'
        }
        patterns['functional_equation'] = {
            'keywords': ['functional equation', 'f(x)', 'substitute', 'evaluate at'],
            'category': 'Algebraic',
            'description': 'Solve functional equations'
        }

        # Structural Methods
        patterns['mathematical_induction'] = {
            'keywords': ['induction', 'base case', 'inductive step', 'assume for n'],
            'category': 'Structural',
            'description': 'Prove by mathematical induction'
        }
        patterns['proof_by_contradiction'] = {
            'keywords': ['contradiction', 'assume the opposite', 'suppose not', 'absurd'],
            'category': 'Structural',
            'description': 'Prove by contradiction'
        }
        patterns['case_analysis'] = {
            'keywords': ['case', 'split', 'consider separately', 'wlog'],
            'category': 'Structural',
            'description': 'Perform case-by-case analysis'
        }
        patterns['extremal_principle'] = {
            'keywords': ['minimal', 'maximal', 'smallest', 'largest', 'extremal'],
            'category': 'Structural',
            'description': 'Use extremal principle'
        }
        patterns['invariant'] = {
            'keywords': ['invariant', 'preserve', 'maintain', 'conserved'],
            'category': 'Structural',
            'description': 'Find and exploit invariants'
        }

        return patterns

    def extract_template(self, strategy_text: str) -> Tuple[str, str, str]:
        """
        Extract template from strategy text
        Returns: (template_name, category, description)
        """
        text_lower = strategy_text.lower()

        # Score each template
        scores = {}
        for template_name, pattern in self.template_patterns.items():
            score = sum(1 for kw in pattern['keywords'] if kw in text_lower)
            if score > 0:
                scores[template_name] = score

        # Return best match
        if scores:
            best_template = max(scores.items(), key=lambda x: x[1])[0]
            pattern = self.template_patterns[best_template]
            return best_template, pattern['category'], pattern['description']

        # Fallback: categorize by keywords
        category = self._categorize_by_keywords(text_lower)
        return f'{category.lower()}_other', category, f'Other {category} methods'

    def _categorize_by_keywords(self, text_lower: str) -> str:
        """Categorize by keywords"""
        if any(kw in text_lower for kw in ['coordinate', 'vector', 'complex']):
            return 'Coordinate'
        if any(kw in text_lower for kw in ['angle', 'circle', 'triangle', 'geometric']):
            return 'Geometric'
        if any(kw in text_lower for kw in ['prime', 'modulo', 'gcd', 'divisib']):
            return 'Number Theory'
        if any(kw in text_lower for kw in ['count', 'combination', 'probability']):
            return 'Combinatorial'
        if any(kw in text_lower for kw in ['induction', 'contradiction', 'case']):
            return 'Structural'
        return 'Algebraic'

    def get_all_templates(self) -> List[Tuple[str, str, str]]:
        """Get all predefined templates"""
        return [(name, p['category'], p['description'])
                for name, p in self.template_patterns.items()]


# ============================================================================
# Part 2: Knowledge Graph Construction
# ============================================================================

class StrategyKnowledgeGraph:
    """Build and manage the strategy knowledge graph"""

    def __init__(self):
        self.problems = []
        self.strategies = []
        self.templates = []
        self.template_names = []  # List of unique template names

        # Mappings
        self.problem_to_idx = {}
        self.strategy_to_idx = {}
        self.template_to_idx = {}

        # ✅ FIX: 分离V1和V13的边，避免标签冲突
        # Graph edges - separated by source
        self.problem_strategy_edges_v1 = []   # (problem_idx, strategy_idx, is_correct_v1)
        self.problem_strategy_edges_v13 = []  # (problem_idx, strategy_idx, is_correct_v13)
        # Legacy combined edges (for backward compatibility, will be populated from v1+v13)
        self.problem_strategy_edges = []  # Will contain both, with source tag
        self.strategy_template_edges = []  # (strategy_idx, template_idx)

        # Strategy -> template mapping for retrieval
        self.strategy_to_template = {}

        # ✅ SOLUTION 1: Strategy -> application examples mapping
        # Maps strategy_idx to list of {problem_id, problem_text, snippet, is_correct}
        self.strategy_applications = {}

        # Statistics
        self.topic_error_rates = {}

    def categorize_strategy(self, strategy_text: str) -> str:
        """Legacy method for compatibility"""
        extractor = StrategyTemplateExtractor()
        _, category, _ = extractor.extract_template(strategy_text)
        return category

    def _extract_snippet(self, solution_text: str, strategy_text: str, max_len: int = 300) -> str:
        """
        ✅ SOLUTION 1: Extract a relevant snippet from solution showing strategy application

        Strategy:
        1. Look for sentences containing key terms from strategy
        2. Extract surrounding context (±1 sentence)
        3. Truncate to max_len while preserving sentence boundaries
        """
        if not solution_text or not strategy_text:
            return ""

        # Split solution into sentences (simple approach)
        import re
        sentences = re.split(r'[.!?]\s+', solution_text)

        # Extract key terms from strategy (lowercase for matching)
        strategy_lower = strategy_text.lower()
        key_terms = set()

        # Common math terms to extract
        math_keywords = ['equation', 'formula', 'theorem', 'lemma', 'identity', 'symmetry',
                        'substitute', 'expand', 'factor', 'simplify', 'solve', 'derive',
                        'proof', 'show', 'verify', 'compute', 'calculate', 'apply']

        for word in strategy_lower.split():
            if len(word) > 4 or word in math_keywords:  # Longer words or math keywords
                key_terms.add(word)

        # Find sentences with highest overlap
        best_sentences = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            overlap = sum(1 for term in key_terms if term in sent_lower)
            if overlap > 0:
                best_sentences.append((overlap, i, sent))

        if not best_sentences:
            # No matching sentences, return first few sentences
            snippet = '. '.join(sentences[:2])
            return snippet[:max_len] + ('...' if len(snippet) > max_len else '')

        # Sort by overlap, take top sentence and surrounding context
        best_sentences.sort(reverse=True)
        best_idx = best_sentences[0][1]

        # Get context: best sentence ± 1
        start_idx = max(0, best_idx - 1)
        end_idx = min(len(sentences), best_idx + 2)

        snippet = '. '.join(sentences[start_idx:end_idx])

        # Truncate to max_len
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rsplit(' ', 1)[0] + '...'

        return snippet

    def _compute_strategy_source_preference(self, human_data, model_data):
        """
        ✅ Compare human vs model strategies on each (subject, template) pair

        Logic:
        1. Compute problem-level accuracy for each (subject, template) in human and model data
        2. Prefer the source with higher accuracy; use 'both' if difference < 5%

        Returns: {(subject, template): 'human' | 'model' | 'both'}
        """
        from collections import defaultdict

        extractor = StrategyTemplateExtractor()

        # Track problem-level accuracy for each (subject, template)
        # Each problem counts only once per template
        stats = defaultdict(lambda: {
            'human_correct': 0, 'human_total': 0,
            'model_correct': 0, 'model_total': 0
        })

        # Process human strategies data
        for result in human_data.get('results', []):
            subject = result.get('subject', 'unknown')
            # ✅ FIX: 统一 subject 为小写，确保与 retriever 一致
            subject = subject.lower()
            is_correct = result.get('judgment', {}).get('is_correct', False)
            strategies = result.get('extracted_theorems', [])

            # 提取这道题用的所有template（去重）
            templates_used = set()
            for strategy in strategies:
                if strategy and isinstance(strategy, str):
                    template_name, _, _ = extractor.extract_template(strategy)
                    templates_used.add(template_name)

            # Count each template once per problem
            for template_name in templates_used:
                key = (subject, template_name)
                stats[key]['human_total'] += 1
                if is_correct:
                    stats[key]['human_correct'] += 1

        # Process model strategies data
        for result in model_data.get('results', []):
            subject = result.get('subject', 'unknown')
            # Normalize subject to lowercase for consistency
            subject = subject.lower()
            is_correct = result.get('judgment', {}).get('is_correct', False)
            strategies = result.get('extracted_theorems', [])

            # Extract all templates used in this problem (deduplicated)
            templates_used = set()
            for strategy in strategies:
                if strategy and isinstance(strategy, str):
                    template_name, _, _ = extractor.extract_template(strategy)
                    templates_used.add(template_name)

            # Count each template once per problem
            for template_name in templates_used:
                key = (subject, template_name)
                stats[key]['model_total'] += 1
                if is_correct:
                    stats[key]['model_correct'] += 1

        # Decide preference by comparing accuracy
        preference = {}
        for key, stat in stats.items():
            human_total = stat['human_total']
            model_total = stat['model_total']

            # Too few samples (<3), use both
            if human_total < 3 and model_total < 3:
                preference[key] = 'both'
                continue

            # Only one source has data
            if human_total < 3:
                preference[key] = 'model'
                continue
            if model_total < 3:
                preference[key] = 'human'
                continue

            # Compute accuracy
            human_acc = stat['human_correct'] / human_total
            model_acc = stat['model_correct'] / model_total

            # Simple rule: if difference < 5%, use both; otherwise prefer higher accuracy
            if abs(human_acc - model_acc) < 0.05:
                preference[key] = 'both'
            elif human_acc > model_acc:
                preference[key] = 'human'
            else:
                preference[key] = 'model'

        print(f"  ✓ Computed preferences for {len(preference)} (subject, template) pairs")

        # 打印统计信息
        human_count = sum(1 for v in preference.values() if v == 'human')
        model_count = sum(1 for v in preference.values() if v == 'model')
        both_count = sum(1 for v in preference.values() if v == 'both')
        print(f"    - Prefer human: {human_count} ({100*human_count/len(preference):.1f}%)")
        print(f"    - Prefer model: {model_count} ({100*model_count/len(preference):.1f}%)")
        print(f"    - Use both: {both_count} ({100*both_count/len(preference):.1f}%)")

        return preference

    def load_data_from_final_dataset(self, dataset_path: str,
                                      human_path: str = None, model_path: str = None):
        """
        Load data from strategy_dataset_final.json

        Args:
            dataset_path: Path to strategy_dataset_final.json
            human_path: Optional path to human strategy results for computing source preference
            model_path: Optional path to model strategy results for computing source preference
        """
        print(f"Loading data from {dataset_path}...")

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Pre-compute source preference if human/model data provided
        if human_path and model_path:
            print("\n  Computing strategy source preferences from human/model data...")
            with open(human_path, 'r') as f:
                human_data = json.load(f)
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            self.strategy_source_preference = self._compute_strategy_source_preference(human_data, model_data)
        else:
            print("\n  ⚠ No preference data provided, strategy_source_preference will not be available")
            self.strategy_source_preference = {}  # Empty dict, retriever will use 'both'

        metadata = data['metadata']
        problems_data = data['data']

        print(f"  Total problems: {metadata['total_problems']}")
        print(f"  Total human strategies: {metadata['total_human_strategies']}")
        print(f"  Total model strategies: {metadata['total_model_strategies']}")

        # Build KG from dataset
        extractor = StrategyTemplateExtractor()
        template_set = set()

        for problem_data in problems_data:
            problem_id = problem_data['problem_id']
            problem_text = problem_data['problem_text']
            # Support both 'subject' and 'problem_subject'
            problem_subject = problem_data.get('subject', problem_data.get('problem_subject', 'unknown'))
            # Support both 'level' and 'problem_level'
            problem_level = problem_data.get('level', problem_data.get('problem_level', 0))
            solution = problem_data.get('solution', '')

            # Add problem node
            if problem_id not in self.problem_to_idx:
                p_idx = len(self.problems)
                self.problem_to_idx[problem_id] = p_idx
                self.problems.append({
                    'problem_id': problem_id,
                    'problem_text': problem_text,
                    'subject': problem_subject,
                    'level': problem_level,
                    'solution': solution,
                    'is_correct_v1': True,  # Assume strategies are correct
                    'is_correct_v13': True
                })
            else:
                p_idx = self.problem_to_idx[problem_id]

            # Process human strategies
            for strategy_data in problem_data.get('human_strategies', []):
                # ✅ FIX: Support both string format (new) and dict format (old)
                if isinstance(strategy_data, str):
                    # New format: simple string, need to extract template
                    strategy_text = strategy_data
                    # Use extractor to classify template (returns tuple)
                    template_name, template_category, template_description = extractor.extract_template(strategy_text)
                else:
                    # Old format: dict with metadata
                    strategy_text = strategy_data['strategy_text']
                    template_name = strategy_data['template_name']
                    template_category = strategy_data['template_category']
                    template_description = strategy_data['template_description']

                if strategy_text not in self.strategy_to_idx:
                    s_idx = len(self.strategies)
                    self.strategy_to_idx[strategy_text] = s_idx
                    self.strategies.append(strategy_text)
                    template_set.add(template_name)

                    self.strategy_to_template[strategy_text] = {
                        'template': template_name,
                        'category': template_category,
                        'description': template_description,
                        'source': 'human'
                    }
                else:
                    s_idx = self.strategy_to_idx[strategy_text]

                # Add edge
                self.problem_strategy_edges_v1.append((p_idx, s_idx, True))

                # Add application example
                snippet = self._extract_snippet(solution, strategy_text, max_len=300)
                if s_idx not in self.strategy_applications:
                    self.strategy_applications[s_idx] = []

                self.strategy_applications[s_idx].append({
                    'problem_id': problem_id,
                    'problem_text': problem_text[:200],
                    'snippet': snippet,
                    'is_correct': True,
                    'source': 'human'
                })

            # Process model strategies
            for strategy_data in problem_data.get('model_strategies', []):
                # ✅ FIX: Support both string format (new) and dict format (old)
                if isinstance(strategy_data, str):
                    # New format: simple string, need to extract template
                    strategy_text = strategy_data
                    # Use extractor to classify template (returns tuple)
                    template_name, template_category, template_description = extractor.extract_template(strategy_text)
                else:
                    # Old format: dict with metadata
                    strategy_text = strategy_data['strategy_text']
                    template_name = strategy_data['template_name']
                    template_category = strategy_data['template_category']
                    template_description = strategy_data['template_description']

                if strategy_text not in self.strategy_to_idx:
                    s_idx = len(self.strategies)
                    self.strategy_to_idx[strategy_text] = s_idx
                    self.strategies.append(strategy_text)
                    template_set.add(template_name)

                    self.strategy_to_template[strategy_text] = {
                        'template': template_name,
                        'category': template_category,
                        'description': template_description,
                        'source': 'model'
                    }
                else:
                    s_idx = self.strategy_to_idx[strategy_text]

                # Add edge
                self.problem_strategy_edges_v13.append((p_idx, s_idx, True))

                # Add application example
                snippet = self._extract_snippet(solution, strategy_text, max_len=300)
                if s_idx not in self.strategy_applications:
                    self.strategy_applications[s_idx] = []

                self.strategy_applications[s_idx].append({
                    'problem_id': problem_id,
                    'problem_text': problem_text[:200],
                    'snippet': snippet,
                    'is_correct': True,
                    'source': 'model'
                })

        # Build template list
        self.template_names = sorted(list(template_set))
        self.template_to_idx = {t: i for i, t in enumerate(self.template_names)}

        # Build strategy -> template edges
        for strategy, info in self.strategy_to_template.items():
            s_idx = self.strategy_to_idx[strategy]
            t_idx = self.template_to_idx[info['template']]
            self.strategy_template_edges.append((s_idx, t_idx))

        # Build templates list with descriptions
        self.templates = []
        for template_name in self.template_names:
            for strategy, info in self.strategy_to_template.items():
                if info['template'] == template_name:
                    self.templates.append({
                        'name': template_name,
                        'category': info['category'],
                        'description': info['description']
                    })
                    break

        # Merge edge tables
        self.problem_strategy_edges = []
        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v1:
            self.problem_strategy_edges.append((p_idx, s_idx, is_correct, 'v1'))

        v1_pairs = {(p_idx, s_idx) for p_idx, s_idx, _ in self.problem_strategy_edges_v1}
        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v13:
            if (p_idx, s_idx) not in v1_pairs:
                self.problem_strategy_edges.append((p_idx, s_idx, is_correct, 'v13'))

        # Compute statistics
        self._compute_topic_error_rates()
        self._compute_template_statistics()

        print(f"\n✓ Loaded {len(self.problems)} problems")
        print(f"✓ Loaded {len(self.strategies)} unique strategies")
        print(f"  - From human: {sum(1 for info in self.strategy_to_template.values() if info.get('source') == 'human')}")
        print(f"  - From model: {sum(1 for info in self.strategy_to_template.values() if info.get('source') == 'model')}")
        print(f"✓ Extracted {len(self.templates)} templates")
        print(f"✓ Built {len(self.problem_strategy_edges)} problem-strategy edges")
        print(f"✓ Built {len(self.strategy_template_edges)} strategy-template edges")

    def load_data(self, v1_path: str, v13_path: str):
        """Load data from v1 and v13 solutions (legacy method)"""
        print("Loading data...")

        with open(v1_path, 'r') as f:
            v1_data = json.load(f)
        with open(v13_path, 'r') as f:
            v13_data = json.load(f)

        # Step 1: 预统计每个(subject, template)应该用人类还是模型的策略
        print("\n  Step 1/3: Pre-computing strategy source preferences...")
        strategy_source_preference = self._compute_strategy_source_preference(v1_data, v13_data)
        self.strategy_source_preference = strategy_source_preference  # Save to instance

        # Build problem list and strategy vocabulary
        print("\n  Step 2/3: Building knowledge graph...")
        extractor = StrategyTemplateExtractor()
        template_set = set()

        # Process v1 (human solutions)
        for result in v1_data.get('results', []):
            problem_id = result.get('problem_id', '')
            subject = result.get('subject', 'unknown')
            # ✅ FIX: 统一入口规范化 - 数据加载时就 lowercase
            subject = (subject or 'unknown').lower()
            problem_text = result.get('problem_text', '')
            level = result.get('level', 0)

            # 正确提取is_correct
            is_correct_v1 = result.get('judgment', {}).get('is_correct', False)

            # ✅ SOLUTION 1: Get solution text for snippet extraction
            solution_text = result.get('reasoning_content', '')

            if problem_id not in self.problem_to_idx:
                self.problem_to_idx[problem_id] = len(self.problems)
                self.problems.append({
                    'problem_id': problem_id,
                    'problem_text': problem_text,
                    'subject': subject,
                    'level': level,
                    'is_correct_v1': is_correct_v1,
                    'is_correct_v13': False
                })

            # 从extracted_theorems提取策略（保留所有策略，不在build时过滤）
            strategies = result.get('extracted_theorems', [])
            for strategy in strategies:
                if not strategy or not isinstance(strategy, str):
                    continue

                # 获取template
                template_name, category, description = extractor.extract_template(strategy)

                # ✅ FIX: 保留所有策略，不要根据preference过滤
                # 过滤应该在推理时做，不是在建图时做

                if strategy not in self.strategy_to_idx:
                    self.strategy_to_idx[strategy] = len(self.strategies)
                    self.strategies.append(strategy)

                    template_set.add(template_name)

                    # ✅ SUPPLEMENT D: 初始化时不立即赋值source，稍后用边表投票
                    # Store mapping (暂不赋值source)
                    self.strategy_to_template[strategy] = {
                        'template': template_name,
                        'category': category,
                        'description': description,
                        'source': 'unknown'  # 稍后根据边表投票
                    }

                s_idx = self.strategy_to_idx[strategy]
                p_idx = self.problem_to_idx[problem_id]
                # ✅ FIX: 分别存储V1和V13的边
                self.problem_strategy_edges_v1.append((p_idx, s_idx, is_correct_v1))

                # ✅ SOLUTION 1: Extract and store snippet showing strategy application
                snippet = self._extract_snippet(solution_text, strategy, max_len=300)
                if s_idx not in self.strategy_applications:
                    self.strategy_applications[s_idx] = []

                self.strategy_applications[s_idx].append({
                    'problem_id': problem_id,
                    'problem_text': problem_text[:200],  # First 200 chars for context
                    'snippet': snippet,
                    'is_correct': is_correct_v1,
                    'source': 'human'
                })

        # Process v13 (model solutions)
        for result in v13_data.get('results', []):
            problem_id = result.get('problem_id', '')

            # 正确提取is_correct
            is_correct_v13 = result.get('judgment', {}).get('is_correct', False)

            # ✅ SOLUTION 1: Get solution text for snippet extraction
            solution_text = result.get('reasoning_content', '')
            problem_text = result.get('problem_text', '')

            if problem_id in self.problem_to_idx:
                p_idx = self.problem_to_idx[problem_id]
                self.problems[p_idx]['is_correct_v13'] = is_correct_v13

            # 从extracted_theorems提取策略（保留所有策略，不在build时过滤）
            subject = self.problems[self.problem_to_idx[problem_id]]['subject'] if problem_id in self.problem_to_idx else 'unknown'

            strategies = result.get('extracted_theorems', [])
            for strategy in strategies:
                if not strategy or not isinstance(strategy, str):
                    continue

                # 获取template
                template_name, category, description = extractor.extract_template(strategy)

                # ✅ FIX: 保留所有策略，不要根据preference过滤
                # 过滤应该在推理时做，不是在建图时做

                if strategy not in self.strategy_to_idx:
                    self.strategy_to_idx[strategy] = len(self.strategies)
                    self.strategies.append(strategy)

                    template_set.add(template_name)

                    # ✅ SUPPLEMENT D: 初始化时不立即赋值source，稍后用边表投票
                    # Store mapping (暂不赋值source)
                    self.strategy_to_template[strategy] = {
                        'template': template_name,
                        'category': category,
                        'description': description,
                        'source': 'unknown'  # 稍后根据边表投票
                    }

                s_idx = self.strategy_to_idx[strategy]
                if problem_id in self.problem_to_idx:
                    p_idx = self.problem_to_idx[problem_id]
                    # ✅ FIX: 分别存储V13的边
                    self.problem_strategy_edges_v13.append((p_idx, s_idx, is_correct_v13))

                    # ✅ SOLUTION 1: Extract and store snippet showing strategy application
                    snippet = self._extract_snippet(solution_text, strategy, max_len=300)
                    if s_idx not in self.strategy_applications:
                        self.strategy_applications[s_idx] = []

                    self.strategy_applications[s_idx].append({
                        'problem_id': problem_id,
                        'problem_text': problem_text[:200],  # First 200 chars for context
                        'snippet': snippet,
                        'is_correct': is_correct_v13,
                        'source': 'model'
                    })

        # Build template list
        self.template_names = sorted(list(template_set))
        self.template_to_idx = {t: i for i, t in enumerate(self.template_names)}

        # Build strategy -> template edges
        for strategy, info in self.strategy_to_template.items():
            s_idx = self.strategy_to_idx[strategy]
            t_idx = self.template_to_idx[info['template']]
            self.strategy_template_edges.append((s_idx, t_idx))

        # Build templates list with descriptions
        self.templates = []
        for template_name in self.template_names:
            # Find first strategy using this template
            example_strategy = None
            for strategy, info in self.strategy_to_template.items():
                if info['template'] == template_name:
                    self.templates.append({
                        'name': template_name,
                        'category': info['category'],
                        'description': info['description']
                    })
                    break

        # ✅ SUPPLEMENT D: 根据边表投票确定每个策略的source
        print("\n  Inferring strategy sources from edge voting...")
        from collections import Counter

        strategy_edge_sources = defaultdict(list)  # s_idx -> list of sources

        # 收集每个策略在哪些边中出现
        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v1:
            strategy_edge_sources[s_idx].append('human')

        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v13:
            strategy_edge_sources[s_idx].append('model')

        # 投票决定source（多数原则）
        inferred_sources = {}
        for s_idx, sources in strategy_edge_sources.items():
            counter = Counter(sources)
            # 选择出现次数最多的source
            most_common_source, count = counter.most_common(1)[0]
            inferred_sources[s_idx] = most_common_source

        # 更新strategy_to_template中的source
        for strategy, s_idx in self.strategy_to_idx.items():
            if s_idx in inferred_sources:
                self.strategy_to_template[strategy]['source'] = inferred_sources[s_idx]

        # 统计source分布
        source_counts = Counter(inferred_sources.values())
        unknown_count = sum(1 for info in self.strategy_to_template.values() if info.get('source') == 'unknown')

        print(f"  ✓ Inferred sources for {len(inferred_sources)} strategies")
        print(f"    - Human: {source_counts.get('human', 0)} ({100*source_counts.get('human', 0)/len(self.strategies):.1f}%)")
        print(f"    - Model: {source_counts.get('model', 0)} ({100*source_counts.get('model', 0)/len(self.strategies):.1f}%)")
        print(f"    - Unknown: {unknown_count} ({100*unknown_count/len(self.strategies):.1f}%)")

        # ✅ FIX: 合并边表用于backward compatibility和统计
        # 选择主要边表：根据preference选择，或默认用V1
        print("\n  Merging edge tables...")
        self.problem_strategy_edges = []

        # 优先使用V1的边（human策略通常更可靠）
        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v1:
            self.problem_strategy_edges.append((p_idx, s_idx, is_correct, 'v1'))

        # 添加V13的边（但如果同一个(p,s)已存在于V1，跳过以避免冲突）
        v1_pairs = {(p_idx, s_idx) for p_idx, s_idx, _ in self.problem_strategy_edges_v1}
        for p_idx, s_idx, is_correct in self.problem_strategy_edges_v13:
            if (p_idx, s_idx) not in v1_pairs:
                self.problem_strategy_edges.append((p_idx, s_idx, is_correct, 'v13'))

        print(f"  ✓ V1 edges: {len(self.problem_strategy_edges_v1)}")
        print(f"  ✓ V13 edges: {len(self.problem_strategy_edges_v13)}")
        print(f"  ✓ Merged edges (de-duplicated): {len(self.problem_strategy_edges)}")

        # Compute statistics
        print("\n  Step 3/3: Computing statistics...")
        self._compute_topic_error_rates()
        self._compute_template_statistics()

        # 统计策略来源
        human_strategies = sum(1 for info in self.strategy_to_template.values() if info.get('source') == 'human')
        model_strategies = sum(1 for info in self.strategy_to_template.values() if info.get('source') == 'model')

        # ✅ SOLUTION 1: 统计应用示例
        total_applications = sum(len(apps) for apps in self.strategy_applications.values())
        strategies_with_apps = len(self.strategy_applications)
        avg_apps_per_strategy = total_applications / max(strategies_with_apps, 1)

        print(f"\n✓ Loaded {len(self.problems)} problems")
        print(f"✓ Loaded {len(self.strategies)} unique strategies")
        print(f"  - From human solutions: {human_strategies} ({100*human_strategies/len(self.strategies):.1f}%)")
        print(f"  - From model solutions: {model_strategies} ({100*model_strategies/len(self.strategies):.1f}%)")
        print(f"✓ Extracted {len(self.templates)} templates")
        print(f"✓ Built {len(self.problem_strategy_edges)} problem-strategy edges")
        print(f"✓ Built {len(self.strategy_template_edges)} strategy-template edges")
        print(f"✓ Collected {total_applications} application examples for {strategies_with_apps} strategies ({avg_apps_per_strategy:.1f} per strategy)")

    def _compute_topic_error_rates(self):
        """Compute error rate for each subject"""
        topic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for problem in self.problems:
            subject = problem['subject']
            is_correct = problem.get('is_correct_v13', False)

            topic_stats[subject]['total'] += 1
            if is_correct:
                topic_stats[subject]['correct'] += 1

        for subject, stats in topic_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                self.topic_error_rates[subject] = 1.0 - accuracy
            else:
                self.topic_error_rates[subject] = 0.0

    def _compute_template_statistics(self):
        """Compute template statistics for (subject, template) pairs"""
        from collections import defaultdict

        # 统计 (subject, template) 的成功率
        # 不使用lambda，直接用字典避免pickle问题
        self.template_stats = {}

        # 统计每个问题使用的策略
        for problem in self.problems:
            p_idx = self.problem_to_idx[problem['problem_id']]
            subject = problem['subject']
            # ✅ FIX: 统一 subject 为小写
            subject = subject.lower()
            is_correct_v1 = problem.get('is_correct_v1', False)
            is_correct_v13 = problem.get('is_correct_v13', False)

            # ✅ FIX C: 按source分别统计v1和v13，避免标签污染
            for edge in self.problem_strategy_edges:
                edge_p_idx, s_idx, is_correct, source = edge[0], edge[1], edge[2], edge[3]

                if edge_p_idx != p_idx:
                    continue

                strategy = self.strategies[s_idx]
                template_name = self.strategy_to_template[strategy]['template']
                key = (subject, template_name)

                # 初始化key（如果不存在）
                if key not in self.template_stats:
                    self.template_stats[key] = {
                        'count_v1': 0, 'correct_v1': 0,
                        'count_v13': 0, 'correct_v13': 0,
                        'total_count': 0
                    }

                # 按source分别统计
                if source == 'v1':
                    self.template_stats[key]['count_v1'] += 1
                    self.template_stats[key]['correct_v1'] += int(is_correct)
                elif source == 'v13':
                    self.template_stats[key]['count_v13'] += 1
                    self.template_stats[key]['correct_v13'] += int(is_correct)

                self.template_stats[key]['total_count'] += 1


# ============================================================================
# Part 3: GNN with Contrastive Learning
# ============================================================================

class ContrastiveGNN(nn.Module):
    """GNN model trained with contrastive learning"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projections
        self.problem_proj = nn.Linear(input_dim, hidden_dim)
        self.strategy_proj = nn.Linear(input_dim, hidden_dim)
        self.template_proj = nn.Linear(input_dim, hidden_dim)

        # Graph transformer layers
        self.conv_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, edge_dim=1)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """Forward pass"""
        # Project inputs
        h_problem = self.problem_proj(x_dict['problem'])
        h_strategy = self.strategy_proj(x_dict['strategy'])
        h_template = self.template_proj(x_dict['template'])

        # Combine
        num_problems = h_problem.size(0)
        num_strategies = h_strategy.size(0)
        num_templates = h_template.size(0)

        h = torch.cat([h_problem, h_strategy, h_template], dim=0)

        # Build unified edge index
        edge_indices = []
        edge_attrs = []

        # Problem -> Strategy edges
        if ('problem', 'uses', 'strategy') in edge_index_dict:
            ps_edges = edge_index_dict[('problem', 'uses', 'strategy')].clone()
            ps_edges[1] += num_problems
            edge_indices.append(ps_edges)
            edge_attrs.append(edge_attr_dict[('problem', 'uses', 'strategy')])

        # Strategy -> Template edges
        if ('strategy', 'belongs_to', 'template') in edge_index_dict:
            st_edges = edge_index_dict[('strategy', 'belongs_to', 'template')].clone()
            st_edges[0] += num_problems
            st_edges[1] += num_problems + num_strategies
            edge_indices.append(st_edges)
            edge_attrs.append(torch.ones(st_edges.size(1), 1, device=st_edges.device))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_attr = torch.cat(edge_attrs, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)
            edge_attr = torch.empty((0, 1), device=h.device)

        # GNN layers
        for i, conv in enumerate(self.conv_layers):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # Residual

        # Output projection
        h = self.output_proj(h)

        # Split back
        h_problem_out = h[:num_problems]
        h_strategy_out = h[num_problems:num_problems + num_strategies]
        h_template_out = h[num_problems + num_strategies:]

        return {
            'problem': h_problem_out,
            'strategy': h_strategy_out,
            'template': h_template_out
        }


def contrastive_loss(anchor, positive, negative, temperature=0.07):
    """InfoNCE contrastive loss"""
    # Normalize
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)

    # Similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature

    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


# ============================================================================
# Part 4: Semantic Embedder
# ============================================================================

class SemanticEmbedder:
    """Encode text using SentenceTransformer"""

    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        print(f"Loading SentenceTransformer: {model_name}")
        # ✅ FIX F: 确保模型在正确的device上
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def encode_problems(self, problems: List[Dict]) -> np.ndarray:
        """Encode problem texts"""
        texts = [p['problem_text'] for p in problems]
        return self.model.encode(texts, show_progress_bar=False)

    def encode_strategies(self, strategies: List[str]) -> np.ndarray:
        """Encode strategy texts"""
        return self.model.encode(strategies, show_progress_bar=False)

    def encode_categories(self, categories: List[str]) -> np.ndarray:
        """Encode category names"""
        return self.model.encode(categories, show_progress_bar=False)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.model.encode([text])[0]


# ============================================================================
# Part 5: Strategy Retriever
# ============================================================================

class StrategyRetriever:
    """Retrieve top-k strategies for a problem"""

    def __init__(self, kg: StrategyKnowledgeGraph, semantic_embedder: SemanticEmbedder,
                 gnn_model: ContrastiveGNN, device: str = 'cpu'):
        self.kg = kg
        self.semantic_embedder = semantic_embedder
        self.gnn_model = gnn_model
        self.device = device

        # Precomputed embeddings (will be set externally)
        self.problem_semantic_emb = None
        self.strategy_semantic_emb = None
        self.template_semantic_emb = None

        self.problem_structural_emb = None
        self.strategy_structural_emb = None
        self.template_structural_emb = None

    def retrieve_topk(self, problem_text: str, k: int = 10,
                     semantic_weight: float = 0.5, structural_weight: float = 0.5,
                     filter_subject: str = None, filter_category: str = None,
                     use_structural: bool = True):
        """
        Retrieve top-k strategies

        Args:
            problem_text: Problem description
            k: Number of strategies to retrieve
            semantic_weight: Weight for semantic similarity
            structural_weight: Weight for structural similarity
            filter_subject: Only retrieve strategies from this subject
            filter_category: Only retrieve strategies from this category
            use_structural: Whether to use structural embeddings
        """
        # Encode problem
        problem_emb_semantic = self.semantic_embedder.encode_text(problem_text)

        # ✅ FIX H: L2 normalize 新问题的 semantic embedding
        problem_emb_semantic = problem_emb_semantic / (np.linalg.norm(problem_emb_semantic) + 1e-8)

        # ✅ FIX G: 使用 kNN 迁移获取 structural embedding（不使用单纯的线性投影）
        # 这部分建议使用 advanced_retriever.py 中的 kNN 方法
        # 为简化，这里暂时禁用 structural（或保持原实现）
        if use_structural and self.gnn_model is not None:
            with torch.no_grad():
                problem_tensor = torch.FloatTensor(problem_emb_semantic).unsqueeze(0).to(self.device)
                problem_emb_structural = self.gnn_model.problem_proj(problem_tensor).squeeze(0).cpu().numpy()
                # ✅ FIX H: L2 normalize structural embedding
                problem_emb_structural = problem_emb_structural / (np.linalg.norm(problem_emb_structural) + 1e-8)
        else:
            problem_emb_structural = problem_emb_semantic

        # Filter strategies by subject/category
        candidate_indices = list(range(len(self.kg.strategies)))

        if filter_subject:
            # Only keep strategies used by problems in the same subject
            subject_strategies = set()
            # ✅ FIX A: edges now have 4 elements (p_idx, s_idx, is_correct, source)
            for edge in self.kg.problem_strategy_edges:
                p_idx, s_idx = edge[0], edge[1]
                if self.kg.problems[p_idx]['subject'] == filter_subject:
                    subject_strategies.add(s_idx)
            candidate_indices = [i for i in candidate_indices if i in subject_strategies]

        if filter_category:
            # Only keep strategies belonging to this category
            category_strategies = set()
            for strategy, info in self.kg.strategy_to_template.items():
                if info['category'] == filter_category:
                    s_idx = self.kg.strategy_to_idx[strategy]
                    category_strategies.add(s_idx)
            candidate_indices = [i for i in candidate_indices if i in category_strategies]

        if not candidate_indices:
            return []

        # Compute similarities
        semantic_scores = np.dot(self.strategy_semantic_emb[candidate_indices], problem_emb_semantic)

        if use_structural and self.strategy_structural_emb is not None:
            structural_scores = np.dot(self.strategy_structural_emb[candidate_indices], problem_emb_structural)
        else:
            structural_scores = np.zeros_like(semantic_scores)

        # Normalize scores
        if semantic_scores.std() > 0:
            semantic_scores = (semantic_scores - semantic_scores.mean()) / semantic_scores.std()
        if use_structural and structural_scores.std() > 0:
            structural_scores = (structural_scores - structural_scores.mean()) / structural_scores.std()

        # Combine scores
        if use_structural:
            combined_scores = semantic_weight * semantic_scores + structural_weight * structural_scores
        else:
            combined_scores = semantic_scores

        # Get top-k
        top_k_local_indices = np.argsort(combined_scores)[::-1][:k]
        top_k_global_indices = [candidate_indices[i] for i in top_k_local_indices]
        top_k_scores = combined_scores[top_k_local_indices]

        results = []
        for idx, score in zip(top_k_global_indices, top_k_scores):
            results.append((idx, float(score)))

        return results


# ============================================================================
# Part 6: Classifier
# ============================================================================

class StrategySuitabilityClassifier:
    """Classify whether a strategy is suitable for a problem"""

    def __init__(self):
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.kg = None
        self.problem_structural_emb = None
        self.strategy_structural_emb = None
        self.template_structural_emb = None
        self.template_semantic_emb = None

    def extract_features(self, problem_idx: int, strategy_idx: int,
                        problem_semantic: np.ndarray, strategy_semantic: np.ndarray) -> np.ndarray:
        """Extract enhanced features for classification"""
        features = []

        problem = self.kg.problems[problem_idx]
        strategy = self.kg.strategies[strategy_idx]
        subject = problem['subject']
        level = problem.get('level', 5) / 10.0  # 归一化到[0,1]

        # 获取template信息
        template_info = self.kg.strategy_to_template.get(strategy, {})
        template_name = template_info.get('template', 'unknown')
        category = template_info.get('category', 'unknown')

        # === 1. 相似度特征 (Similarity Features) ===

        # 1.1 语义相似度 (cosine)
        sem_ps = np.dot(problem_semantic, strategy_semantic) / (
            np.linalg.norm(problem_semantic) * np.linalg.norm(strategy_semantic) + 1e-8
        )
        features.append(sem_ps)

        # 1.2 结构相似度 (来自GNN)
        if self.problem_structural_emb is not None and self.strategy_structural_emb is not None:
            problem_struct = self.problem_structural_emb[problem_idx]
            strategy_struct = self.strategy_structural_emb[strategy_idx]
            struct_ps = np.dot(problem_struct, strategy_struct) / (
                np.linalg.norm(problem_struct) * np.linalg.norm(strategy_struct) + 1e-8
            )
            features.append(struct_ps)
        else:
            features.append(0.0)

        # 1.3 问题-模板相似度 (结构)
        if (self.problem_structural_emb is not None and
            self.template_structural_emb is not None and
            template_name in self.kg.template_to_idx):
            t_idx = self.kg.template_to_idx[template_name]
            problem_struct = self.problem_structural_emb[problem_idx]
            template_struct = self.template_structural_emb[t_idx]
            struct_pt = np.dot(problem_struct, template_struct) / (
                np.linalg.norm(problem_struct) * np.linalg.norm(template_struct) + 1e-8
            )
            features.append(struct_pt)
        else:
            features.append(0.0)

        # 1.4 问题-模板相似度 (语义)
        if (self.template_semantic_emb is not None and
            template_name in self.kg.template_to_idx):
            t_idx = self.kg.template_to_idx[template_name]
            template_sem = self.template_semantic_emb[t_idx]
            sem_pt = np.dot(problem_semantic, template_sem) / (
                np.linalg.norm(problem_semantic) * np.linalg.norm(template_sem) + 1e-8
            )
            features.append(sem_pt)
        else:
            features.append(0.0)

        # === 2. 统计/先验特征 (Statistical Features) ===

        # 2.1 Subject级别错误率
        topic_err = self.kg.topic_error_rates.get(subject, 0.5)
        features.append(topic_err)

        # 2.2 (Subject, Template) 成功率
        key = (subject, template_name)
        stats = self.kg.template_stats.get(key, {
            'count_v1': 0, 'correct_v1': 0,
            'count_v13': 0, 'correct_v13': 0
        })

        # Human accuracy for this (subject, template)
        temp_human_acc = stats['correct_v1'] / max(stats['count_v1'], 1)
        features.append(temp_human_acc)

        # Model accuracy for this (subject, template)
        temp_model_acc = stats['correct_v13'] / max(stats['count_v13'], 1)
        features.append(temp_model_acc)

        # Accuracy gap (human better? model better?)
        temp_acc_gap = temp_human_acc - temp_model_acc
        features.append(temp_acc_gap)

        # 2.3 Template频率特征
        freq_human = np.log1p(stats['count_v1'])  # log(1 + count)
        freq_model = np.log1p(stats['count_v13'])
        freq_total = np.log1p(stats['count_v1'] + stats['count_v13'])
        features.append(freq_human)
        features.append(freq_model)
        features.append(freq_total)

        # === 3. 任务兼容性特征 (Task Compatibility) ===

        # 3.1 Subject-Category一致性
        # 定义期望的category映射
        subject_to_category = {
            'number_theory': 'Number Theory',
            'geometry': 'Geometric',
            'algebra': 'Algebraic',
            'counting_and_probability': 'Combinatorial',
            'precalculus': 'Algebraic',
        }
        expected_cat = subject_to_category.get(subject, category)
        is_same_category = 1.0 if expected_cat == category else 0.0
        features.append(is_same_category)

        # 3.2 题目难度归一化
        features.append(level)

        # 3.3 模板在该难度级别的平均表现（简化版：模板总体成功率）
        template_overall_acc = (stats['correct_v1'] + stats['correct_v13']) / max(
            stats['count_v1'] + stats['count_v13'], 1
        )
        features.append(template_overall_acc)

        return np.array(features)

    def train(self, kg: StrategyKnowledgeGraph,
             problem_semantic_emb: np.ndarray,
             strategy_semantic_emb: np.ndarray,
             problem_structural_emb: np.ndarray = None,
             strategy_structural_emb: np.ndarray = None,
             template_structural_emb: np.ndarray = None,
             template_semantic_emb: np.ndarray = None):
        """Train the classifier with enhanced features"""
        self.kg = kg
        self.problem_structural_emb = problem_structural_emb
        self.strategy_structural_emb = strategy_structural_emb
        self.template_structural_emb = template_structural_emb
        self.template_semantic_emb = template_semantic_emb

        X = []
        y = []

        # ✅ FIX: edges now have 4 elements (p_idx, s_idx, is_correct, source)
        for edge in kg.problem_strategy_edges:
            p_idx, s_idx, is_correct = edge[0], edge[1], edge[2]
            features = self.extract_features(
                p_idx, s_idx,
                problem_semantic_emb[p_idx],
                strategy_semantic_emb[s_idx]
            )
            X.append(features)
            y.append(1 if is_correct else 0)

        X = np.array(X)
        y = np.array(y)

        print(f"  Training with {X.shape[0]} examples, {X.shape[1]} features")

        # Check if we have both positive and negative examples
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"  ⚠ Warning: Only {unique_classes} class found in training data")
            print(f"  ⚠ Skipping classifier training (requires both positive and negative examples)")
            print(f"  ⚠ Classifier will use default prediction of 0.5")
            self.classifier = None  # Mark as untrained
        else:
            self.classifier.fit(X, y)
            print(f"✓ Trained classifier")

        return self

    def predict(self, problem_idx: int, strategy_idx: int,
               problem_semantic: np.ndarray, strategy_semantic: np.ndarray) -> float:
        """Predict suitability probability"""
        if self.classifier is None:
            # Classifier not trained, return neutral probability
            return 0.5
        features = self.extract_features(problem_idx, strategy_idx, problem_semantic, strategy_semantic)
        prob = self.classifier.predict_proba([features])[0][1]
        return prob


if __name__ == '__main__':
    print("Strategy Knowledge Graph System")
    print("Use build_kg.py or other scripts to run specific tasks")
