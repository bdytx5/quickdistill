# Generate Test Traces

This script generates test traces using multiple models on the housing_qa legal Q&A dataset.

## Setup

1. **Install dependencies:**
```bash
pip install datasets
```

2. **Set environment variables:**
```bash
export WANDB_API_KEY="your_wandb_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # Optional
```

Get OpenRouter key at: https://openrouter.ai/keys

## Run

```bash
python generate_test_traces.py
```

## What it does

1. Loads 10 questions from the `reglab/housing_qa` dataset
2. Tests each question on 6 different models:
   - **W&B Inference** (3 models):
     - Llama-3.3-70B
     - DeepSeek-V3.1
     - Qwen3-235B
   - **OpenRouter** (3 models):
     - Claude 3.5 Sonnet
     - Gemini 2.0 Flash (free)
     - Llama 3.3 70B (free)

3. All runs are logged to Weave project: `byyoung3/test-housing-qa`

## View Results

1. Open `trace_viewer.html`
2. Enter project name: `byyoung3/test-housing-qa`
3. Click "Fetch New Project"
4. Filter by model/operation and select traces
5. Export selected traces to create test sets
6. Run weak models on the exported test set
7. Evaluate with judges

## Customize

Edit `generate_test_traces.py` to:
- Change `NUM_QUESTIONS` (default: 10)
- Add/remove models from `WANDB_MODELS` or `OPENROUTER_MODELS`
- Change the Weave project name in `PROJECT_NAME`
