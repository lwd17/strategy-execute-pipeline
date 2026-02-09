# Dataset

The full strategy dataset (`strategy_dataset_full.json`, ~41 MB) is not included in this repository.

## Download

Download from one of these sources:

1. **Hugging Face** (recommended):
   ```bash
   # TODO: Add download link after uploading to Hugging Face
   ```

2. **Google Drive**:
   ```bash
   # TODO: Add Google Drive link
   ```

3. **Manual placement**:
   - Place your own `strategy_dataset_full.json` in this directory
   - See [README.md](../README.md) for expected format

## File Structure

After downloading, this directory should contain:
```
data/
├── README.md (this file)
└── strategy_dataset_full.json (4850 problems, ~41 MB)
```

## Dataset Format

The dataset should follow one of these formats (see main README for details):

**Format 1: Simple strings**
```json
{
  "metadata": {...},
  "data": [
    {
      "problem_id": "...",
      "problem_text": "...",
      "subject": "...",
      "human_strategies": ["strategy text..."],
      "model_strategies": ["strategy text..."]
    }
  ]
}
```

**Format 2: Objects with metadata** (hybrid_kg_system format)
```json
{
  "data": [
    {
      "problem_id": "...",
      "human_strategies": [
        {
          "strategy_text": "...",
          "template_name": "...",
          "template_category": "...",
          "template_description": "..."
        }
      ]
    }
  ]
}
```
