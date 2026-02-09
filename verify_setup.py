#!/usr/bin/env python3
"""
验证系统配置是否正确
"""
import os
import sys

def main():
    print("=" * 70)
    print("验证 Math Strategy Pipeline 配置")
    print("=" * 70)

    all_good = True

    # 1. 检查环境变量
    print("\n[1/5] 检查环境变量...")
    vllm_model = os.environ.get("VLLM_MODEL")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if vllm_model:
        print(f"  ✓ VLLM_MODEL = {vllm_model}")
    else:
        print(f"  ⚠ VLLM_MODEL 未设置（将使用默认: Qwen/Qwen3-8B）")

    if openai_key:
        print(f"  ✓ OPENAI_API_KEY = {openai_key[:10]}...")
    else:
        print(f"  ✗ OPENAI_API_KEY 未设置！")
        all_good = False

    # 2. 检查预训练文件
    print("\n[2/5] 检查预训练文件...")
    required_files = [
        'strategy_kg.pkl',
        'gnn_model.pth',
        'strategy_classifier.pkl',
        'problem_semantic_emb.npy',
        'problem_structural_emb.npy',
        'strategy_semantic_emb.npy',
        'strategy_structural_emb.npy',
        'template_semantic_emb.npy',
        'template_structural_emb.npy'
    ]

    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} 缺失")
            missing.append(f)
            all_good = False

    if missing:
        print(f"\n  缺失 {len(missing)} 个文件！")

    # 3. 检查数据文件
    print("\n[3/5] 检查数据文件...")
    if os.path.exists('data/strategy_dataset_full.json'):
        size_mb = os.path.getsize('data/strategy_dataset_full.json') / 1024 / 1024
        print(f"  ✓ data/strategy_dataset_full.json ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ data/strategy_dataset_full.json 缺失")
        all_good = False

    # 4. 检查Python模块
    print("\n[4/5] 检查Python模块...")
    sys.path.insert(0, 'src')
    try:
        from retriever_v2 import OptimizedStrategyRetriever
        print(f"  ✓ retriever_v2 可导入")
    except Exception as e:
        print(f"  ✗ retriever_v2 导入失败: {e}")
        all_good = False

    try:
        from strategy_kg import StrategyKnowledgeGraph
        print(f"  ✓ strategy_kg 可导入")
    except Exception as e:
        print(f"  ✗ strategy_kg 导入失败: {e}")
        all_good = False

    # 5. 检查vLLM服务器
    print("\n[5/5] 检查vLLM服务器...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print(f"  ✓ vLLM服务器运行中")
        else:
            print(f"  ⚠ vLLM服务器响应异常: {response.status_code}")
    except:
        print(f"  ✗ vLLM服务器未运行")
        print(f"    请运行: bash scripts/start_vllm_server.sh")
        all_good = False

    # 总结
    print("\n" + "=" * 70)
    if all_good:
        print("✅ 所有检查通过！系统已准备就绪")
        print("=" * 70)
        print("\n现在可以运行:")
        print("  python tests/test_aime25_accuracy.py")
        print("  python tests/test_apex_accuracy.py")
        return 0
    else:
        print("❌ 发现问题，请修复后再运行测试")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
