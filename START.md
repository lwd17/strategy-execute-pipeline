# Quick Start - 3 Steps

## Step 1: Start vLLM Server

**Terminal 1** - Start server:
```bash
cd math_strategy_pipeline
bash scripts/start_vllm_server.sh
```

Select model (option 1 recommended):
```
1) Qwen/Qwen3-8B      ← Recommended
2) Qwen/Qwen3-14B
3) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Wait for: `Application startup complete`

**Remember the exact model name**, e.g.: `Qwen/Qwen3-8B`

---

## Step 2: Configure Environment

**Terminal 2** - Configure and test:
```bash
cd math_strategy_pipeline

# Set model name (must match Step 1!)
export VLLM_MODEL="Qwen/Qwen3-8B"

# Set OpenAI API Key
export OPENAI_API_KEY="sk-..."

# Check which KG is loaded
python check_kg.py

# Verify setup
python verify_setup.py
```

Expected output: `✅ All checks passed!`

---

## Step 3: Run Tests

```bash
# Test AIME 2025 (30 problems)
python tests/test_aime25_accuracy.py

# Or test APEX
python tests/test_apex_accuracy.py
```

Results saved to:
- `aime25_results.json`
- `apex_shortlist_results_v2.json`

---

## Troubleshooting

### Q: `NameError: name 'VLLM_MODEL' is not defined`
**A**: Environment variable not set, re-run export command in Step 2

### Q: All scores are 0
**A**: Check vLLM server is running: `curl http://localhost:8000/health`

### Q: Model name mismatch
**A**: Ensure environment variable matches server:
```bash
# Check server log
tail logs/vllm_server.log | grep "model="

# Check environment
echo $VLLM_MODEL

# Must match exactly!
```

---

## Supported Models

| Model | Name String |
|-------|-------------|
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Qwen3-14B | `Qwen/Qwen3-14B` |
| DeepSeek-R1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

---

**Ready to start!** 🚀
