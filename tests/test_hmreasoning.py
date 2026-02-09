#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 100-problem test set with train set KG retrieval

This script:
- Loads 100 test problems (from strategy_dataset_test.json)
- Uses TrainSetRetriever to retrieve strategies (from 4832 train problems only)
- Uses Qwen3-8B (vLLM) to solve problems with strategy guidance
- Uses GPT-5.1 to judge results and calculate accuracy
- Ensures NO data leakage (test problems never appear in retrieved strategies)
"""

import os
import sys
import json
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as tqdm_asyncio

# Add current directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retriever_train import TrainSetRetriever

try:
    from openai import AsyncOpenAI
except Exception:
    raise RuntimeError("Please install openai>=1.30.0: pip install 'openai>=1.30.0'")


# ---------------------------
# Config
# ---------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
VLLM_MAX_TOKENS = 32768
VLLM_TEMPERATURE = 0.7

# GPT-5.1 for judging
OPENAI_MODEL_JUDGE = os.environ.get("OPENAI_MODEL_JUDGE", "gpt-5.1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

CACHE_DIR = "./cache_test_split"
MAX_OPENAI_CONC = 10

os.makedirs(CACHE_DIR, exist_ok=True)

# Output file
OUTPUT_FILE = './test_split_results.json'


# ---------------------------
# Helper functions
# ---------------------------
def sha1_jsonable(obj):
    """Compute SHA1 hash of JSON-serializable object"""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def cache_path(prefix: str, key: str) -> str:
    """Get cache file path"""
    return os.path.join(CACHE_DIR, f"{prefix}_{key}.json")


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text"""
    import re
    text = re.sub(r'^```(?:json)?\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------
# Retriever
# ---------------------------
class StrategyRetriever:
    def __init__(self):
        print("Initializing Train Set Strategy Retriever...")
        self.retriever = TrainSetRetriever()
        print("✓ Retriever ready (using train set KG only)\n")

    def retrieve_strategies(self, problem_text: str, k: int = 5, subject: str = None, preferred_source: str = 'auto') -> List[Dict]:
        """Retrieve top-k strategies for a problem"""
        return self.retriever.retrieve(
            problem_text=problem_text,
            subject=subject,
            k=k,
            top_templates=2,
            top_similar_problems=5,
            semantic_weight=0.5,
            structural_weight=0.5,
            diversity_control=True,
            ann_top=50,
            preferred_source=preferred_source
        )


# ---------------------------
# Solver
# ---------------------------
class StrategySolver:
    def __init__(self, vllm_base_url: str, model: str):
        self.vllm_base_url = vllm_base_url
        self.model = model

    def build_prompt(self, problem_text: str, strategies: List[Dict]) -> List[Dict]:
        """Build prompt with retrieved strategies"""
        if strategies:
            strategy_lines = []
            for i, strat in enumerate(strategies, 1):
                strategy_lines.append(
                    f"{i}. [{strat['template']}] {strat['strategy']}"
                )
            strategies_text = "\n".join(strategy_lines)

            user_content = f"""Solve this problem:

{problem_text}

Similar problems were solved using these approaches:
{strategies_text}

Based on these strategies, solve the problem step by step."""
        else:
            user_content = f"""Solve the following mathematical problem step by step.

Problem:
{problem_text}

Provide a detailed solution with clear reasoning and the final answer."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert mathematician solving competition problems."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        return messages

    async def solve_single(self, session: aiohttp.ClientSession,
                          problem_text: str, strategies: List[Dict],
                          idx: int) -> Dict[str, str]:
        """Solve a single problem with strategies"""
        messages = self.build_prompt(problem_text, strategies)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": VLLM_MAX_TOKENS,
            "temperature": VLLM_TEMPERATURE,
        }

        try:
            async with session.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                timeout=None
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {
                        "reasoning_content": "",
                        "content": f"API error {resp.status}: {text[:200]}"
                    }

                result = await resp.json()
                message = result['choices'][0]['message']

                # Extract reasoning content from DeepSeek-R1 format
                # vLLM omits the opening <think> tag, only outputs </think>
                full_content = message.get('content') or ''
                reasoning_content = ''
                content = full_content

                if '</think>' in full_content:
                    # Split at </think> tag
                    parts = full_content.split('</think>', 1)
                    if len(parts) == 2:
                        # Everything before </think> is reasoning
                        reasoning_content = parts[0].strip()
                        # Remove opening <think> tag if present
                        if reasoning_content.startswith('<think>'):
                            reasoning_content = reasoning_content[7:].strip()
                        # Everything after </think> is the final answer
                        content = parts[1].strip()

                # Fallback: if there's a reasoning_content field, use it
                if not reasoning_content and message.get('reasoning_content'):
                    reasoning_content = message.get('reasoning_content')

                # Retry if empty
                if not content:
                    max_retries = 3
                    for retry_num in range(1, max_retries + 1):
                        print(f"  Problem {idx+1}: ⚠ Empty content, retry {retry_num}/{max_retries}...")
                        async with session.post(
                            f"{self.vllm_base_url}/chat/completions",
                            json=payload,
                            timeout=None
                        ) as retry_resp:
                            if retry_resp.status == 200:
                                retry_result = await retry_resp.json()
                                retry_message = retry_result['choices'][0]['message']

                                # Extract reasoning content from DeepSeek-R1 format
                                retry_full_content = retry_message.get('content') or ''
                                reasoning_content = ''
                                content = retry_full_content

                                if '</think>' in retry_full_content:
                                    retry_parts = retry_full_content.split('</think>', 1)
                                    if len(retry_parts) == 2:
                                        reasoning_content = retry_parts[0].strip()
                                        if reasoning_content.startswith('<think>'):
                                            reasoning_content = reasoning_content[7:].strip()
                                        content = retry_parts[1].strip()

                                if not reasoning_content and retry_message.get('reasoning_content'):
                                    reasoning_content = retry_message.get('reasoning_content')

                                if content:
                                    print(f"  Problem {idx+1}: ✓ Retry {retry_num} succeeded")
                                    break

                print(f"  Problem {idx+1}: ✓ Generated (reasoning: {len(reasoning_content)} chars, content: {len(content)} chars)")
                return {
                    "reasoning_content": reasoning_content,
                    "content": content
                }

        except Exception as e:
            print(f"  Problem {idx+1}: ✗ Error: {str(e)[:120]}")
            return {
                "reasoning_content": "",
                "content": f"Error: {str(e)[:200]}"
            }

    async def solve_batch(self, problems: List[Dict],
                         strategies_list: List[List[Dict]]) -> List[Dict]:
        """Solve a batch of problems"""
        print(f"\nSolving {len(problems)} problems with {VLLM_MODEL}...")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, (prob, strategies) in enumerate(zip(problems, strategies_list)):
                task = self.solve_single(session, prob['problem_text'], strategies, i)
                tasks.append(task)

            results = await tqdm_asyncio.gather(*tasks, desc="Solving problems")
            return results


# ---------------------------
# Judge (GPT-5.1)
# ---------------------------
class GPTJudge:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._sem = asyncio.Semaphore(MAX_OPENAI_CONC)

    async def judge_single(self, problem: str, student_solution: Dict[str, str],
                          reference_answer: str, idx: int) -> Dict[str, Any]:
        """Judge a single answer using GPT-5.1"""
        student_final = student_solution.get('content', '').strip()

        prompt = f"""You are a rigorous mathematics expert evaluating student answers.

Problem:
{problem}

Ground Truth Answer:
{reference_answer}

Student's Final Answer:
{student_final}

Evaluation Task:
Compare the student's final answer with the ground truth answer and determine if they are mathematically equivalent.

Mathematical Equivalence Rules:
- Account for different valid representations:
  * Fractions vs decimals (e.g., 1/2 = 0.5)
  * Mixed numbers vs improper fractions
  * Simplified vs unsimplified forms
  * Different algebraic forms
- Be STRICT about numerical values - if numbers don't match, it's WRONG
- Be STRICT about signs - negative vs positive matters

Classification:
- "Completely Correct": Final answer is mathematically equivalent to ground truth
- "Completely Wrong": Final answer is not equivalent, or missing, or nonsensical

Return your evaluation in JSON format:
{{
    "category": "Completely Correct" | "Completely Wrong",
    "is_correct": true | false,
    "score": score (0-100, where 100 = completely correct, 0 = completely wrong),
    "explanation": "brief explanation of your evaluation"
}}"""

        try:
            async with self._sem:
                resp = await self.client.chat.completions.create(
                    model=OPENAI_MODEL_JUDGE,
                    messages=[
                        {"role": "system", "content": "You are a rigorous mathematics expert who evaluates student solutions with strict standards."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=1024,
                    temperature=0.3
                )

            result_text = resp.choices[0].message.content.strip()
            result_text = strip_code_fences(result_text)
            result = json.loads(result_text)

            category = result.get("category", "Completely Wrong")
            is_correct = category == "Completely Correct"

            return {
                "category": category,
                "is_correct": is_correct,
                "score": int(result.get("score", 0)),
                "explanation": str(result.get("explanation", "")),
                "student_final_answer": student_final
            }

        except Exception as e:
            print(f"  Problem {idx+1}: ✗ Judge error: {str(e)[:100]}")
            return {
                "category": "Completely Wrong",
                "is_correct": False,
                "score": 0,
                "explanation": f"Error: {str(e)[:100]}",
                "student_final_answer": student_final
            }

    async def judge_batch(self, problems: List[Dict], student_solutions: List[Dict]) -> List[Dict]:
        """Judge a batch of solutions"""
        print(f"\nJudging {len(problems)} solutions with {OPENAI_MODEL_JUDGE}...")

        tasks = []
        for i, (prob, solution) in enumerate(zip(problems, student_solutions)):
            task = self.judge_single(
                problem=prob['problem_text'],
                student_solution=solution,
                reference_answer=prob['solution'],
                idx=i
            )
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks, desc="Judging solutions")
        return results


# ---------------------------
# Main workflow
# ---------------------------
async def main():
    print("="*70)
    print("TEST SET EVALUATION (Train Set KG Retrieval)")
    print("="*70)

    # 1. Load test set
    print(f"\n[1/5] Loading test set (100 problems)...")

    test_data_path = './strategy_dataset_test.json'
    if not os.path.exists(test_data_path):
        print(f"✗ Test set not found: {test_data_path}")
        print("  Please run split_dataset.py first!")
        return

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    problems = []
    for item in test_data['data']:
        problems.append({
            'problem_id': item['problem_id'],
            'problem_text': item['problem_text'],
            'problem_subject': item.get('problem_subject', 'unknown'),
            'problem_level': item.get('problem_level', 'unknown'),
            'solution': item['solution'],
            'solution_source': item.get('solution_source', 'unknown')
        })

    print(f"✓ Loaded {len(problems)} test problems")
    print(f"  Random seed used for split: {test_data['metadata'].get('random_seed', 'unknown')}")

    # 2. Initialize retriever
    print(f"\n[2/5] Initializing train set retriever...")
    retriever = StrategyRetriever()

    # 3. Retrieve strategies for each problem
    print(f"\n[3/5] Retrieving strategies for each test problem...")
    strategies_list = []
    for i, prob in enumerate(problems):
        print(f"  Retrieving strategies for problem {i+1}/{len(problems)}...")
        strategies = retriever.retrieve_strategies(
            prob['problem_text'],
            k=7,
            subject=prob['problem_subject']
        )
        strategies_list.append(strategies)
        print(f"    Retrieved {len(strategies)} strategies from train set")

    # 4. Solve problems with strategies
    print(f"\n[4/5] Solving problems with {VLLM_MODEL}...")
    solver = StrategySolver(VLLM_BASE_URL, VLLM_MODEL)
    solutions = await solver.solve_batch(problems, strategies_list)

    # 5. Judge with GPT-5.1
    print(f"\n[5/5] Judging solutions with {OPENAI_MODEL_JUDGE}...")
    judge = GPTJudge()
    judgments = await judge.judge_batch(problems, solutions)

    # Calculate accuracy
    results = []
    correct_count = 0

    for i, (prob, strategies, solution, judgment) in enumerate(zip(problems, strategies_list, solutions, judgments)):
        is_correct = judgment['is_correct']

        if is_correct:
            correct_count += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  {status} Problem {i+1}: {judgment['category']} (score: {judgment['score']})")

        # Build the prompt that was used
        if strategies:
            strategy_lines = []
            for j, strat in enumerate(strategies, 1):
                strategy_lines.append(f"{j}. [{strat['template']}] {strat['strategy']}")
            strategies_text = "\n".join(strategy_lines)
            prompt_text = f"""Solve this problem:

{prob['problem_text']}

Similar problems were solved using these approaches:
{strategies_text}

Based on these strategies, solve the problem step by step."""
        else:
            prompt_text = f"""Solve the following mathematical problem step by step.

Problem:
{prob['problem_text']}

Provide a detailed solution with clear reasoning and the final answer."""

        result = {
            "problem_id": prob['problem_id'],
            "problem_text": prob['problem_text'],
            "problem_subject": prob['problem_subject'],
            "problem_level": prob['problem_level'],
            "ground_truth_answer": prob['solution'],
            "solution_source": prob['solution_source'],

            # Extracted strategies
            "retrieved_strategies": [
                {
                    "template": s['template'],
                    "category": s['category'],
                    "source": s['source'],
                    "score": s['score'],
                    "strategy_text": s['strategy']
                }
                for s in strategies
            ],
            "strategies_count": len(strategies),

            # Prompt used
            "prompt": prompt_text,

            # Model outputs
            "reasoning_content": solution['reasoning_content'],
            "final_answer": solution['content'],

            # Judgment
            "judgment": {
                "is_correct": judgment['is_correct'],
                "category": judgment['category'],
                "score": judgment['score'],
                "explanation": judgment['explanation'],
                "predicted_answer_extracted": judgment['student_final_answer']
            }
        }
        results.append(result)

    # Calculate metrics
    accuracy = correct_count / len(problems) * 100 if problems else 0.0
    avg_score = sum(j['judgment']['score'] for j in results) / len(results) if results else 0.0
    avg_strategies = sum(j['strategies_count'] for j in results) / len(results) if results else 0.0

    # Category breakdown
    category_counts = {
        "Completely Correct": sum(1 for j in results if j['judgment']['category'] == "Completely Correct"),
        "Completely Wrong": sum(1 for j in results if j['judgment']['category'] == "Completely Wrong")
    }

    # Save results
    output = {
        "statistics": {
            "total_problems": len(problems),
            "correct": correct_count,
            "incorrect": len(problems) - correct_count,
            "accuracy": round(accuracy, 2),
            "average_score": round(avg_score, 2),
            "category_breakdown": category_counts,
            "average_strategies_per_problem": round(avg_strategies, 1),
            "model": VLLM_MODEL,
            "judge_model": OPENAI_MODEL_JUDGE,
            "retriever": "TrainSetRetriever (4832 train problems, 0 test problems)",
            "test_set_size": len(problems),
            "train_set_size": 4832,
            "random_seed": test_data['metadata'].get('random_seed', 'unknown')
        },
        "results": results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Test set: {len(problems)} problems")
    print(f"Train set: 4832 problems (used for KG)")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count}/{len(problems)} = {accuracy:.2f}%")
    print(f"Average score: {avg_score:.2f}")
    print(f"Average strategies per problem: {avg_strategies:.1f}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())