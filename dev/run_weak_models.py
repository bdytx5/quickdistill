import os
import json
import openai
import weave
from pathlib import Path

# Configuration
PROJECT = "wandb_inference"
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.5",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1-0528",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
]

weave.init(PROJECT)

def create_client():
    """Create and return OpenAI client configured for W&B Inference."""
    return openai.OpenAI(
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.getenv("WANDB_API_KEY"),
        project=PROJECT,
        default_headers={
            "OpenAI-Project": "wandb_fc/quickstart_playground"  # replace with your team/project
        }
    )

def run_inference(client, model, messages, max_tokens=1000):
    """Run inference with specified model and parameters."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # Using temperature 0 as requested
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def load_strong_model_data():
    """Load the strong model traces from strong_model.json"""
    strong_model_path = Path("strong_model.json")
    if not strong_model_path.exists():
        print("ERROR: strong_model.json not found!")
        print("Please use the trace viewer UI to select and export traces first.")
        return None

    with open(strong_model_path, 'r') as f:
        return json.load(f)

def extract_messages_from_trace(trace):
    """Extract messages from a trace in the format needed for inference"""
    # Check if messages are at top level
    if trace.get('messages') and isinstance(trace['messages'], list) and len(trace['messages']) > 0:
        return trace['messages']

    # Check if messages are in inputs
    if trace.get('inputs') and isinstance(trace['inputs'], dict):
        messages = trace['inputs'].get('messages', [])
        if isinstance(messages, list) and len(messages) > 0:
            return messages

    return None

def main():
    # Load strong model data
    print("Loading strong_model.json...")
    strong_traces = load_strong_model_data()
    if not strong_traces:
        return

    print(f"Loaded {len(strong_traces)} traces from strong_model.json\n")

    # Display available models
    print("Available models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")

    # Get model selection
    print("\nEnter model numbers to run (comma-separated, e.g., 1,3,5) or 'all':")
    selection = input("> ").strip()

    if selection.lower() == 'all':
        selected_models = AVAILABLE_MODELS
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_models = [AVAILABLE_MODELS[i] for i in indices if 0 <= i < len(AVAILABLE_MODELS)]
        except:
            print("Invalid selection!")
            return

    if not selected_models:
        print("No models selected!")
        return

    # Get number of examples to generate
    print("\nHow many examples to generate per model?")
    print(f"(Max available: {len(strong_traces)})")
    num_examples_input = input("> ").strip()

    try:
        num_examples = int(num_examples_input)
        if num_examples <= 0:
            print("Number must be positive!")
            return
        num_examples = min(num_examples, len(strong_traces))
    except:
        print("Invalid number!")
        return

    print(f"\nWill run {num_examples} examples through {len(selected_models)} model(s)\n")
    print("=" * 80)

    # Create client
    client = create_client()

    # Run inference for each model
    for model in selected_models:
        print(f"\nRunning model: {model}")
        print("-" * 80)

        results = []

        for i, trace in enumerate(strong_traces[:num_examples]):
            print(f"Processing example {i+1}/{num_examples}...", end=' ')

            # Extract messages
            messages = extract_messages_from_trace(trace)

            if not messages:
                print("SKIP (no messages found)")
                results.append({
                    "trace_id": trace.get('id'),
                    "messages": None,
                    "output": None,
                    "error": "No messages found in trace"
                })
                continue

            # Run inference
            output = run_inference(client, model, messages)

            # Store result
            result = {
                "trace_id": trace.get('id'),
                "messages": messages,
                "output": output,
                "strong_model": trace.get('model'),
                "strong_model_output": trace.get('output')
            }
            results.append(result)

            print("DONE")

        # Save results
        model_safe_name = model.replace('/', '_')
        output_file = f"weak_model_{model_safe_name}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved {len(results)} results to {output_file}")
        print("=" * 80)

    print("\nAll done!")

if __name__ == "__main__":
    main()
