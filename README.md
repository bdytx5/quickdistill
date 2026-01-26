# QuickDistill

**Fast and easy toolkit for evaluating AI models (and training them in the future)**

QuickDistill provides an intuitive web UI for the complete model distillation workflow:
- 📊 View and filter Weave traces from your LLM calls
- 🎯 Export strong model outputs as test sets
- 🔬 Run weak models on strong outputs
- ⚖️ Evaluate similarity using LLM judges
- 📥 Download evaluation datasets for analysis

## Installation

```bash
pip install quickdistill
```

## Quick Start

Launch the UI in your current directory:

```bash
quickdistill launch
```

This will:
- Start the QuickDistill server on `http://localhost:5001`
- Open the Trace Viewer in your browser
- Create local directories for projects, exports, and results

## Requirements

Set these environment variables:

```bash
export WANDB_API_KEY="your_wandb_key"          # Required for W&B Inference
export OPENROUTER_API_KEY="your_openrouter_key"  # Optional for OpenRouter models
```

Get your keys:
- W&B: https://wandb.ai/settings
- OpenRouter: https://openrouter.ai/keys

## Usage

### 1. Fetch Weave Traces

Enter your Weave project name (e.g., `username/project-name`) and click "Fetch New Project" to load traces from W&B.

### 2. Export Strong Model Outputs

- Filter traces by model or operation
- Select traces to use as ground truth
- Export to create a test set

### 3. Run Weak Models

- Select a strong model export
- Choose W&B models from the list or enter custom OpenRouter models
- Run inference to generate weak model outputs

### 4. Create Judges

Navigate to `/judge` to create LLM judges:
- **Scalar judges**: Rate similarity on a numeric scale (1-5)
- **Boolean judges**: Determine if outputs are correct/incorrect

### 5. Run Evaluations

- Select weak model results
- Choose a judge
- Run evaluation and view results in Weave

### 6. Download Datasets

Click "Download" next to any weak model result to get a clean JSON dataset with:
```json
[
  {
    "input": "question text...",
    "strong_model": "model-name",
    "strong_output": "strong response...",
    "weak_model": "model-name",
    "weak_output": "weak response..."
  }
]
```

## CLI Options

```bash
quickdistill launch                    # Launch on default port 5001
quickdistill launch --port 8080        # Launch on custom port
quickdistill launch --no-browser       # Don't open browser automatically
quickdistill launch --debug            # Run in debug mode
```

## Project Structure

When you run `quickdistill launch`, it creates these directories ~/.cache/quickdistill:

```
your-project/
├── projects/                  # Cached Weave traces by project
│   └── username_project/
│       └── traces_data.json
├── strong_exports/            # Exported strong model test sets
│   └── model-name_10traces.json
├── weak_model_*.json          # Weak model inference results
├── judges.json                # Saved judge configurations
└── evaluations/               # Evaluation results
```

## Features

- **Multi-provider support**: Works with W&B Inference and OpenRouter
- **Flexible judging**: Create custom LLM judges or use pre-built ones
- **Trace filtering**: Filter by model, operation, or custom criteria
- **Batch operations**: Run multiple models and evaluations in parallel
- **Export formats**: Download clean datasets for further analysis
- **Project isolation**: Each Weave project is cached separately

## Development

Install in development mode:

```bash
git clone https://github.com/byyoung3/quickdistill.git
cd quickdistill
pip install -e .
```

## License

MIT

## Author

Brett Young (bdytx5@umsystem.edu)
