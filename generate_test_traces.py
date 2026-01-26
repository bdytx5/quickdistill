"""
Generate test traces using different models on the financial-qa-10K dataset
"""
import os
import weave
import openai
from datasets import load_dataset

# Initialize weave
PROJECT_NAME = "byyoung3/test-financial-qa"
weave.init(PROJECT_NAME)

# W&B Inference models
WANDB_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]

# OpenRouter models
OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

NUM_QUESTIONS = 10  # Number of questions to test


def create_wandb_client():
    """Create W&B Inference client"""
    return openai.OpenAI(
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.getenv("WANDB_API_KEY"),
        project=PROJECT_NAME,
        default_headers={
            "OpenAI-Project": "wandb_fc/quickstart_playground"
        }
    )


def create_openrouter_client():
    """Create OpenRouter client"""
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )


@weave.op()
def run_wandb_inference(client, model: str, question: str, context: str = None) -> str:
    """Run inference using W&B Inference"""
    if context:
        prompt = f"""Based on the following context, answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
    else:
        prompt = f"""Answer the following question concisely based on your knowledge.

Question: {question}

Answer:"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with {model}: {e}")
        return f"ERROR: {str(e)}"


@weave.op()
def run_openrouter_inference(client, model: str, question: str, context: str = None) -> str:
    """Run inference using OpenRouter"""
    if context:
        prompt = f"""Based on the following context, answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
    else:
        prompt = f"""Answer the following question concisely based on your knowledge.

Question: {question}

Answer:"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with {model}: {e}")
        return f"ERROR: {str(e)}"


def main():
    print(f"Generating test traces for project: {PROJECT_NAME}")
    print(f"Loading {NUM_QUESTIONS} questions from financial-qa-10K dataset...")

    # Load dataset - financial-qa-10K is in parquet format
    dataset = load_dataset("virattt/financial-qa-10K", split="train")

    # Take first NUM_QUESTIONS
    questions = dataset.select(range(min(NUM_QUESTIONS, len(dataset))))

    print(f"\nLoaded {len(questions)} questions")
    print(f"Testing {len(WANDB_MODELS)} W&B models and {len(OPENROUTER_MODELS)} OpenRouter models")
    print("=" * 80)

    # Create clients
    wandb_client = create_wandb_client()
    openrouter_client = create_openrouter_client()

    # Run W&B models
    for model in WANDB_MODELS:
        print(f"\nRunning W&B model: {model}")
        print("-" * 80)

        for i, sample in enumerate(questions):
            question = sample['question']
            context = sample.get('context', '')
            correct_answer = sample['answer']

            print(f"  [{i+1}/{len(questions)}] {question[:60]}...", end=" ")

            result = run_wandb_inference(
                wandb_client,
                model,
                question,
                context
            )

            # Check if answer is in the result
            match = "✓" if correct_answer.lower() in result.lower() else "✗"
            print(f"{match}")
            print(f"    Answer: {result[:80]}...")

    # Run OpenRouter models
    for model in OPENROUTER_MODELS:
        print(f"\nRunning OpenRouter model: {model}")
        print("-" * 80)

        for i, sample in enumerate(questions):
            question = sample['question']
            context = sample.get('context', '')
            correct_answer = sample['answer']

            print(f"  [{i+1}/{len(questions)}] {question[:60]}...", end=" ")

            result = run_openrouter_inference(
                openrouter_client,
                model,
                question,
                context
            )

            # Check if answer is in the result
            match = "✓" if correct_answer.lower() in result.lower() else "✗"
            print(f"{match}")
            print(f"    Answer: {result[:80]}...")

    print("\n" + "=" * 80)
    print(f"✓ Done! All traces logged to Weave project: {PROJECT_NAME}")
    print(f"View at: https://wandb.ai/{PROJECT_NAME}")
    print("\nNext steps:")
    print("1. Open trace_viewer.html")
    print(f"2. Fetch traces for project: {PROJECT_NAME}")
    print("3. Select traces and export them to create a test set")


if __name__ == "__main__":
    # Check for required env vars
    if not os.getenv("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY environment variable not set")
        exit(1)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not set - OpenRouter models will be skipped")
        print("Get your key at: https://openrouter.ai/keys")
        OPENROUTER_MODELS.clear()

    main()
