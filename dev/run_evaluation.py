import json
import weave
import os
import openai
from pathlib import Path

from typing import Dict, Any, List

# Initialize weave
weave.init("auto-distill-eval")


def load_judges():
    """Load judge definitions from localStorage export"""
    judge_file = Path("judges.json")
    if not judge_file.exists():
        print("ERROR: judges.json not found!")
        print("Please export judges from the Judge Manager UI")
        return []

    with open(judge_file, 'r') as f:
        return json.load(f)


def load_weak_model_results(model_file: str):
    """Load weak model results from JSON file"""
    if not Path(model_file).exists():
        print(f"ERROR: {model_file} not found!")
        return []

    with open(model_file, 'r') as f:
        return json.load(f)


def create_llm_client():
    """Create OpenAI client for judge LLM calls"""
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_llm_judge(judge: Dict[str, Any], strong_output: str, weak_output: str, question: str = "") -> Dict[str, float]:
    """Run LLM-as-a-judge evaluation"""
    client = create_llm_client()

    # Format prompt using replace to avoid issues with literal curly braces
    prompt = judge['prompt']
    prompt = prompt.replace('{question}', question)
    prompt = prompt.replace('{strong_output}', strong_output)
    prompt = prompt.replace('{weak_output}', weak_output)

    # Call LLM
    try:
        response = client.chat.completions.create(
            model=judge['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Error running LLM judge: {e}")
        return {}


def run_custom_judge(judge: Dict[str, Any], strong_output: str, weak_output: str) -> Dict[str, float]:
    """Run custom function judge"""
    # Execute custom function
    try:
        # Create a namespace for execution
        namespace = {}
        exec(judge['customFunction'], namespace)

        # Get the judge function
        judge_func = namespace.get('custom_judge')
        if not judge_func:
            print("ERROR: custom_judge function not found in custom code")
            return {}

        # Run the judge
        result = judge_func(strong_output, weak_output)
        return result

    except Exception as e:
        print(f"Error running custom judge: {e}")
        return {}


def evaluate_model(weak_model_file: str, judge_name: str = None):
    """
    Run evaluation on weak model results using specified judge

    Args:
        weak_model_file: Path to weak model results JSON
        judge_name: Name of judge to use (if None, user will select)
    """
    # Load judges
    judges = load_judges()
    if not judges:
        return

    # Select judge
    if judge_name:
        judge = next((j for j in judges if j['name'] == judge_name), None)
        if not judge:
            print(f"ERROR: Judge '{judge_name}' not found")
            return
    else:
        print("Available judges:")
        for i, j in enumerate(judges, 1):
            print(f"  {i}. {j['name']} ({j['type']})")
        selection = int(input("Select judge number: ")) - 1
        judge = judges[selection]

    print(f"\nUsing judge: {judge['name']}")

    # Load weak model results
    results = load_weak_model_results(weak_model_file)
    if not results:
        return

    print(f"Loaded {len(results)} examples from {weak_model_file}\n")

    # Extract model name from filename
    model_name = Path(weak_model_file).stem.replace('weak_model_', '')

    # Create evaluation logger
    ev = weave.EvaluationLogger(
        name=f"eval-{model_name}-{judge['name']}",
        model=model_name
    )

    # Run evaluation on each example
    for i, example in enumerate(results):
        print(f"Evaluating example {i+1}/{len(results)}...", end=' ')

        strong_output = example.get('strong_model_output', '')
        weak_output = example.get('output', '')

        # Extract question from messages if available
        question = ""
        messages = example.get('messages', [])
        if messages and len(messages) > 0:
            question = messages[0].get('content', '')

        # Run judge
        if judge['type'] == 'llm':
            scores = run_llm_judge(judge, strong_output, weak_output, question)
        else:
            scores = run_custom_judge(judge, strong_output, weak_output)

        # Log to weave
        ev.log_example(
            inputs={
                "question": question,
                "strong_output": strong_output,
                "weak_output": weak_output
            },
            output=weak_output,
            scores=scores
        )

        print("DONE")

    # Finish evaluation
    ev.log_summary()

    print(f"\n✓ Evaluation complete!")
    print(f"Results URL: {ev.ui_url}")


def main():
    """Interactive evaluation runner"""
    print("=== Auto-Distill Evaluation Runner ===\n")

    # Find weak model result files
    weak_model_files = list(Path(".").glob("weak_model_*.json"))

    if not weak_model_files:
        print("ERROR: No weak model result files found!")
        print("Run weak model inference first using the UI")
        return

    print("Available weak model results:")
    for i, f in enumerate(weak_model_files, 1):
        print(f"  {i}. {f.name}")

    selection = int(input("\nSelect file number: ")) - 1
    selected_file = weak_model_files[selection]

    # Run evaluation
    evaluate_model(str(selected_file))


if __name__ == "__main__":
    main()
