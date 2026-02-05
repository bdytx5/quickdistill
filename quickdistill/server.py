import os
import json
import openai
import weave
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from llmasajudge import LLMAsAJudge
from pathlib import Path

# Get the package directory
PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / 'static'

# Universal data directory
DATA_DIR = Path.home() / '.cache' / 'quickdistill'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(DATA_DIR / 'projects').mkdir(exist_ok=True)
(DATA_DIR / 'strong_exports').mkdir(exist_ok=True)

# Copy default project if it doesn't exist
default_project_src = PACKAGE_DIR / 'default_projects' / 'byyoung3_arena-detailed'
default_project_dst = DATA_DIR / 'projects' / 'byyoung3_arena-detailed'
if default_project_src.exists() and not default_project_dst.exists():
    shutil.copytree(default_project_src, default_project_dst)
    print(f"📦 Installed default project: byyoung3/arena-detailed")

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# Progress tracking for long-running operations
progress_state = {}

# Load settings
SETTINGS_FILE = DATA_DIR / 'settings.json'
DEFAULT_SETTINGS = {
    'inference_project': 'wandb_fc/quickstart_playground',
    'evaluation_project': 'wandb_inference'
}

def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, 'r') as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

settings = load_settings()
PROJECT = settings['evaluation_project']

weave.init(PROJECT)

def create_client():
    """Create and return OpenAI client configured for W&B Inference."""
    return openai.OpenAI(
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.getenv("WANDB_API_KEY"),
        project=PROJECT,
        default_headers={
            "OpenAI-Project": settings['inference_project']
        }
    )

def create_openrouter_client():
    """Create and return OpenAI client configured for OpenRouter."""
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

def get_client_for_model(model):
    """Return appropriate client based on model string."""
    # OpenRouter models typically have these patterns
    openrouter_patterns = ['anthropic/', 'google/', 'openai/gpt-4', 'openai/o1']

    # Check if model matches OpenRouter patterns
    for pattern in openrouter_patterns:
        if pattern in model:
            return create_openrouter_client()

    # Default to W&B Inference
    return create_client()

def run_inference(client, model, messages, max_tokens=1000):
    """Run inference with specified model and parameters."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # Using temperature 0 for deterministic results
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def extract_output_content(output_str):
    """Extract actual content from WeaveObject string, JSON response, or regular output.

    Handles outputs from:
    - OpenAI chat.completions.create (plain text)
    - OpenAI responses.create (JSON with nested structure)
    - Anthropic Messages (WeaveObject with content[0].text)
    - Google Gemini (WeaveObject with candidates[0].content.parts[0].text)
    """
    import re
    import json

    if not output_str:
        return None

    if not isinstance(output_str, str):
        return str(output_str)

    # Handle empty/streaming responses
    if output_str in ('', 'None', 'null'):
        return '[Streaming output - not captured]'

    # Handle OpenAI responses.create JSON format
    if output_str.startswith('{') and '"output"' in output_str:
        try:
            resp_obj = json.loads(output_str)
            if 'output' in resp_obj and isinstance(resp_obj['output'], list):
                # Extract text from output messages
                text_parts = []
                for item in resp_obj['output']:
                    if item.get('type') == 'message' and 'content' in item:
                        for content in item['content']:
                            if content.get('type') == 'output_text' and 'text' in content:
                                text_parts.append(content['text'])
                if text_parts:
                    return '\n\n'.join(text_parts)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Fall through to other handlers

    # Handle WeaveObject strings (Anthropic, Gemini)
    if 'WeaveObject' in output_str:
        # Improved regex that handles escape sequences properly
        match = re.search(r"'text':\s*'((?:[^'\\]|\\.)*)'", output_str, re.DOTALL)
        if match:
            # Unescape the string properly (order matters!)
            text = match.group(1)
            text = text.replace("\\'", "'")      # escaped single quotes
            text = text.replace('\\"', '"')      # escaped double quotes
            text = text.replace('\\n', '\n')     # newlines
            text = text.replace('\\t', '\t')     # tabs
            text = text.replace('\\r', '\r')     # carriage returns
            text = text.replace('\\\\', '\\')    # escaped backslashes (do this last!)
            return text

        # If no text field found, return truncated version
        return f"[Complex WeaveObject - could not extract text]\n{output_str[:500]}..."

    # Plain text output (standard OpenAI chat format)
    return output_str


def extract_messages_from_trace(trace):
    """Extract messages from a trace in the format needed for inference.

    Handles message extraction from:
    - OpenAI chat.completions.create (messages at top level or in inputs.messages)
    - OpenAI responses.create (inputs.input field)
    - Anthropic Messages (inputs.messages)
    - Google Gemini generate_content (inputs.contents array)
    - Google Gemini Chat.send_message (inputs.message string)
    """
    import re

    # Get op_display_name for provider detection
    op_name = trace.get('op_display_name', '')

    # Check if messages are at top level (already extracted/cached)
    if trace.get('messages') and isinstance(trace['messages'], list) and len(trace['messages']) > 0:
        return trace['messages']

    # Check if messages are in inputs
    if trace.get('inputs') and isinstance(trace['inputs'], dict):
        inputs = trace['inputs']

        # Standard OpenAI/Anthropic: inputs.messages
        messages = inputs.get('messages', [])
        if isinstance(messages, list) and len(messages) > 0:
            return messages

        # OpenAI responses.create: inputs.input (simple string)
        if 'openai.responses' in op_name and 'input' in inputs:
            return [{"role": "user", "content": inputs['input']}]

        # Gemini Chat.send_message: inputs.message (simple string)
        if 'Chat.send_message' in op_name and 'message' in inputs:
            return [{"role": "user", "content": inputs['message']}]

        # Gemini generate_content: inputs.contents (array of content objects or WeaveObject strings)
        if 'google.genai' in op_name and 'contents' in inputs:
            contents = inputs['contents']
            if isinstance(contents, list) and len(contents) > 0:
                messages = []
                for content in contents:
                    # Handle WeaveObject string format
                    if isinstance(content, str) and 'WeaveObject' in content:
                        role_match = re.search(r"'role':\s*'(\w+)'", content)
                        text_match = re.search(r"'text':\s*'((?:[^'\\]|\\.)*)'", content, re.DOTALL)
                        text = '[Complex content]'
                        if text_match:
                            text = text_match.group(1)
                            text = text.replace("\\'", "'").replace('\\n', '\n').replace('\\\\', '\\')
                        messages.append({
                            "role": role_match.group(1) if role_match else "user",
                            "content": text
                        })
                    # Handle regular dict format
                    elif isinstance(content, dict):
                        role = content.get('role', 'user')
                        parts = content.get('parts', [])
                        if isinstance(parts, list):
                            text = '\n'.join([p.get('text', '') for p in parts if isinstance(p, dict)])
                            messages.append({"role": role, "content": text})
                if messages:
                    return messages

        # Check if inputs has question/context format (from generate_test_traces.py wrapper traces)
        question = inputs.get('question')
        context = inputs.get('context')
        if question:
            if context:
                prompt = f"""Based on the following context, answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Answer the following question concisely based on your knowledge.

Question: {question}

Answer:"""
            return [{"role": "user", "content": prompt}]

    # Check child_calls if available
    if trace.get('child_calls') and isinstance(trace['child_calls'], list):
        for child in trace['child_calls']:
            # Try to get messages from child
            if child.get('messages') and isinstance(child['messages'], list) and len(child['messages']) > 0:
                return child['messages']
            # Try to get messages from child inputs
            if child.get('inputs') and isinstance(child['inputs'], dict):
                child_messages = child['inputs'].get('messages', [])
                if isinstance(child_messages, list) and len(child_messages) > 0:
                    return child_messages

    return None

@app.route('/run_inference', methods=['POST'])
def run_inference_endpoint():
    """API endpoint to run inference on selected models and traces"""
    data = request.json
    models = data.get('models', [])
    strong_export_file = data.get('strong_export_file')
    num_examples = data.get('num_examples')
    task_id = data.get('task_id', f"inference_{id(models)}")

    if not models:
        return jsonify({'error': 'No models provided'}), 400

    # Load traces from strong export file
    if not strong_export_file:
        return jsonify({'error': 'No strong export file specified'}), 400

    export_path = DATA_DIR / 'strong_exports' / strong_export_file
    if not export_path.exists():
        return jsonify({'error': f'Export file not found: {strong_export_file}'}), 404

    with open(export_path, 'r') as f:
        traces = json.load(f)

    if not traces:
        return jsonify({'error': 'No traces in export file'}), 400

    # Limit traces to num_examples (convert to int if needed)
    if num_examples:
        num_examples = int(num_examples)
        traces = traces[:num_examples]

    output_files = []

    # Initialize progress tracking
    total_steps = len(models) * len(traces)
    progress_state[task_id] = {
        'current': 0,
        'total': total_steps,
        'message': 'Starting inference...',
        'status': 'running'
    }

    # Run inference for each model
    for model_idx, model in enumerate(models):
        print(f"Running model: {model}")
        results = []

        # Get appropriate client for this model
        client = get_client_for_model(model)

        for i, trace in enumerate(traces):
            step = model_idx * len(traces) + i + 1
            progress_state[task_id] = {
                'current': step,
                'total': total_steps,
                'message': f'[{model_idx+1}/{len(models)}] {model} - Example {i+1}/{len(traces)}',
                'status': 'running'
            }
            print(f"  Processing example {i+1}/{len(traces)}...", end=' ')

            # Extract messages
            messages = extract_messages_from_trace(trace)

            if not messages:
                print("SKIP (no messages)")
                results.append({
                    "trace_id": trace.get('id'),
                    "messages": None,
                    "output": None,
                    "error": "No messages found in trace"
                })
                continue

            # Run inference
            output = run_inference(client, model, messages)

            # Extract and clean strong model output
            strong_output = extract_output_content(trace.get('output'))

            # Store result
            result = {
                "trace_id": trace.get('id'),
                "messages": messages,
                "output": output,
                "strong_model": trace.get('model'),
                "strong_model_output": strong_output,
                "strong_export_file": strong_export_file  # Track which export this came from
            }
            results.append(result)

            print("DONE")

        # Save results with metadata
        model_safe_name = model.replace('/', '_')
        output_file = DATA_DIR / f"weak_model_{model_safe_name}.json"

        # Add metadata to results
        output_data = {
            "metadata": {
                "weak_model": model,
                "strong_export_file": strong_export_file,
                "num_examples": len(results)
            },
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        output_files.append(str(output_file))
        print(f"Saved {len(results)} results to {output_file}")

    # Mark progress as complete
    progress_state[task_id] = {
        'current': total_steps,
        'total': total_steps,
        'message': 'Complete!',
        'status': 'complete'
    }

    return jsonify({
        'status': 'success',
        'files': output_files,
        'total_examples': len(traces),
        'models_run': len(models),
        'task_id': task_id
    })

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """Get progress for a running task"""
    if task_id in progress_state:
        return jsonify(progress_state[task_id])
    return jsonify({'error': 'Task not found'}), 404


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify(settings)


@app.route('/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    global settings
    data = request.json
    settings.update(data)
    save_settings(settings)
    return jsonify({'status': 'success', 'settings': settings})


@app.route('/test_judge', methods=['POST'])
def test_judge():
    """Test a judge on sample data to see raw inputs/outputs"""
    data = request.json
    judge = data.get('judge')
    weak_model_file = data.get('weak_model_file')
    num_samples = data.get('num_samples', 5)

    if not judge or not weak_model_file:
        return jsonify({'error': 'Missing judge or weak_model_file'}), 400

    # Load weak model results
    model_path = DATA_DIR / weak_model_file
    with open(model_path, 'r') as f:
        file_data = json.load(f)

    # Handle both formats
    if isinstance(file_data, dict) and 'results' in file_data:
        results = file_data['results']
    else:
        results = file_data

    # Limit to num_samples
    samples_to_test = results[:min(num_samples, len(results))]

    test_results = []

    for example in samples_to_test:
        # Skip examples with errors
        if example.get('error') or not example.get('output'):
            continue

        strong_output = example.get('strong_model_output', '')
        weak_output = example.get('output', '')

        # Extract question
        question = ""
        messages = example.get('messages', [])
        if messages and len(messages) > 0:
            question = messages[0].get('content', '')

        # Build the prompt
        prompt = judge['prompt']
        if '{question}' in prompt:
            prompt = prompt.replace('{question}', question or '')
        if '{strong_output}' in prompt:
            prompt = prompt.replace('{strong_output}', strong_output or '')
        if '{weak_output}' in prompt:
            prompt = prompt.replace('{weak_output}', weak_output or '')

        # Run the judge and capture raw response
        if judge['type'] == 'llm':
            return_type = judge.get('returnType', 'scalar')

            # Use a list to capture the raw response (mutable so we can access from closure)
            captured_raw = []

            def score_parser(response: str):
                """Parse the judge response based on return type"""
                # Capture the raw response before any processing
                captured_raw.append(response)

                response = response.strip()

                # Remove markdown code blocks if present
                if response.startswith('```'):
                    # Remove ```json or ``` at start
                    response = response.split('\n', 1)[1] if '\n' in response else response[3:]
                    # Remove ``` at end
                    if response.endswith('```'):
                        response = response.rsplit('\n', 1)[0] if '\n' in response else response[:-3]
                    response = response.strip()

                try:
                    # Parse JSON response
                    parsed = json.loads(response)

                    if return_type == 'boolean':
                        # Extract boolean value - return just the bool
                        val = parsed.get('correct', parsed.get('result', parsed.get('value', False)))
                        return bool(val)
                    elif return_type == 'scalar':
                        # Extract numeric score - return just the number
                        val = parsed.get('score', parsed.get('scores', parsed.get('value', 0)))
                        return float(val) if isinstance(val, (int, float)) else 0
                    else:
                        # Unsupported return type
                        print(f"Unsupported return type: {return_type}")
                        return 0
                except:
                    print(f"Failed to parse judge response as JSON: {response}")
                    if return_type == 'scalar':
                        return 0
                    elif return_type == 'boolean':
                        return False
                    else:
                        return 0

            # Use LLMAsAJudge exactly like the evaluation code
            try:
                # Initialize LLMAsAJudge with custom prompt
                judge_instance = LLMAsAJudge(
                    models=[judge['model']],
                    use_fully_custom_prompt=True,
                    output_parser=score_parser,
                    return_type=return_type if return_type else None
                )

                # Get judgment
                result = judge_instance.judge(prompt=prompt)

                # Extract the raw response that was captured
                raw_text = captured_raw[0] if captured_raw else "No response captured"

                # Extract parsed scores from result
                if return_type == 'scalar':
                    score_val = result.get('scores', result.get('correct', 0))
                    parsed_scores = {'score': score_val}
                elif return_type == 'boolean':
                    bool_val = result.get('correct', False)
                    parsed_scores = {'correct': bool_val}
                else:
                    # Unsupported return type - default to scalar
                    score_val = result.get('scores', result.get('correct', 0))
                    parsed_scores = {'score': score_val}

            except Exception as e:
                raw_text = f"Error: {str(e)}"
                parsed_scores = {'error': str(e)}

            test_results.append({
                'question': question,
                'strong_output': strong_output,
                'weak_output': weak_output,
                'judge_prompt': prompt,
                'raw_response': raw_text,
                'parsed_scores': parsed_scores
            })

    return jsonify({
        'status': 'success',
        'judge_name': judge['name'],
        'num_samples': len(test_results),
        'samples': test_results
    })


@app.route('/generate_judge_prompt', methods=['POST'])
def generate_judge_prompt():
    """Generate a judge prompt using AI based on sample data"""
    data = request.json
    weak_model_file = data.get('weak_model_file')
    num_samples = data.get('num_samples', 3)
    model = data.get('model', 'openai/gpt-5')
    meta_prompt = data.get('meta_prompt')

    if not weak_model_file or not meta_prompt:
        return jsonify({'error': 'Missing weak_model_file or meta_prompt'}), 400

    # Load weak model results
    model_path = DATA_DIR / weak_model_file
    with open(model_path, 'r') as f:
        file_data = json.load(f)

    # Handle both formats
    if isinstance(file_data, dict) and 'results' in file_data:
        results = file_data['results']
    else:
        results = file_data

    # Limit to num_samples
    samples_to_use = results[:min(num_samples, len(results))]

    # Format samples for meta-prompt
    samples_text = []
    for i, example in enumerate(samples_to_use):
        # Skip examples with errors
        if example.get('error') or not example.get('output'):
            continue

        strong_output = example.get('strong_model_output', '')
        weak_output = example.get('output', '')

        # Extract question
        question = ""
        messages = example.get('messages', [])
        if messages and len(messages) > 0:
            question = messages[0].get('content', '')

        samples_text.append(f"""Sample {i+1}:
Question: {question}
Strong Model Output: {strong_output}
Weak Model Output: {weak_output}
---""")

    samples_formatted = "\n\n".join(samples_text)

    # Replace {SAMPLES} placeholder in meta-prompt
    final_prompt = meta_prompt.replace('{SAMPLES}', samples_formatted)

    # Call OpenRouter to generate the prompt
    try:
        client = create_openrouter_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": final_prompt}]
        )
        generated_prompt = response.choices[0].message.content.strip()

        return jsonify({
            'status': 'success',
            'generated_prompt': generated_prompt,
            'num_samples_used': len(samples_text)
        })

    except Exception as e:
        return jsonify({'error': f'Failed to generate prompt: {str(e)}'}), 500


@app.route('/list_weak_models', methods=['GET'])
def list_weak_models():
    """List available weak model result files with metadata"""
    files_data = []

    for filepath in DATA_DIR.glob("weak_model_*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract metadata
            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']
                files_data.append({
                    'filename': filepath.name,
                    'weak_model': metadata.get('weak_model', 'unknown'),
                    'strong_export': metadata.get('strong_export_file', 'unknown'),
                    'num_examples': metadata.get('num_examples', 0)
                })
            else:
                # Old format - no metadata
                files_data.append({
                    'filename': filepath.name,
                    'weak_model': 'unknown',
                    'strong_export': 'unknown',
                    'num_examples': len(data) if isinstance(data, list) else 0
                })
        except:
            # Failed to read file
            files_data.append({
                'filename': filepath.name,
                'weak_model': 'unknown',
                'strong_export': 'unknown',
                'num_examples': 0
            })

    return jsonify({'files': files_data})


@app.route('/get_weak_model/<filename>', methods=['GET'])
def get_weak_model(filename):
    """Get a specific weak model file"""
    # Security check - ensure filename is just a filename, not a path
    if '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    filepath = DATA_DIR / filename

    if not filepath.exists():
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 500


@app.route('/delete_weak_model/<filename>', methods=['DELETE'])
def delete_weak_model(filename):
    """Delete a specific weak model file"""
    # Security check - ensure filename is just a filename, not a path
    if '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    # Ensure it's a weak_model file
    if not filename.startswith('weak_model_'):
        return jsonify({'error': 'Can only delete weak_model files'}), 400

    filepath = DATA_DIR / filename

    if not filepath.exists():
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        filepath.unlink()  # Delete the file
        model_name = filename.replace('weak_model_', '').replace('.json', '')
        return jsonify({
            'status': 'success',
            'message': f'Successfully deleted evaluation set for {model_name}'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500


@app.route('/export_strong_traces', methods=['POST'])
def export_strong_traces():
    """Export strong model traces to strong_exports/ directory"""
    data = request.json
    traces = data.get('traces', [])
    nickname = data.get('nickname', 'export')

    if not traces:
        return jsonify({'error': 'No traces provided'}), 400

    # Create strong_exports directory if it doesn't exist
    exports_dir = DATA_DIR / 'strong_exports'
    exports_dir.mkdir(exist_ok=True)

    # Generate filename
    filename = f"{nickname}.json"
    filepath = exports_dir / filename

    # Save traces
    with open(filepath, 'w') as f:
        json.dump(traces, f, indent=2)

    return jsonify({
        'status': 'success',
        'filename': filename,
        'path': str(filepath),
        'count': len(traces)
    })


@app.route('/list_strong_exports', methods=['GET'])
def list_strong_exports():
    """List available strong model export files"""
    exports_dir = DATA_DIR / 'strong_exports'
    if not exports_dir.exists():
        return jsonify({'files': []})

    files = [f.name for f in exports_dir.glob("*.json")]
    return jsonify({'files': files})


@app.route('/save_judge', methods=['POST'])
def save_judge():
    """Save a judge definition to judges.json"""
    data = request.json
    judge = data.get('judge')

    if not judge:
        return jsonify({'error': 'No judge provided'}), 400

    # Load existing judges
    judges_file = DATA_DIR / 'judges.json'
    if judges_file.exists():
        with open(judges_file, 'r') as f:
            judges = json.load(f)
    else:
        judges = []

    # Add or update judge
    judge_name = judge.get('name')
    existing_idx = next((i for i, j in enumerate(judges) if j.get('name') == judge_name), None)

    if existing_idx is not None:
        judges[existing_idx] = judge
    else:
        judges.append(judge)

    # Save back
    with open(judges_file, 'w') as f:
        json.dump(judges, f, indent=2)

    return jsonify({'status': 'success', 'judges': judges})


@app.route('/list_judges', methods=['GET'])
def list_judges():
    """List all saved judges (default + user judges)"""
    # Load default judges from package
    default_judges_file = PACKAGE_DIR / 'default_judges.json'
    default_judges = []
    if default_judges_file.exists():
        with open(default_judges_file, 'r') as f:
            default_judges = json.load(f)

    # Load user judges from CWD
    judges_file = DATA_DIR / 'judges.json'
    user_judges = []
    if judges_file.exists():
        with open(judges_file, 'r') as f:
            user_judges = json.load(f)

    # Merge: default judges first, then user judges (user can override)
    judges = default_judges.copy()

    # Add user judges, replacing any defaults with same name
    for user_judge in user_judges:
        existing_idx = next((i for i, j in enumerate(judges) if j.get('name') == user_judge.get('name')), None)
        if existing_idx is not None:
            judges[existing_idx] = user_judge
        else:
            judges.append(user_judge)

    return jsonify({'judges': judges})


@app.route('/delete_judge', methods=['POST'])
def delete_judge():
    """Delete a judge by name"""
    data = request.json
    judge_name = data.get('name')

    if not judge_name:
        return jsonify({'error': 'No judge name provided'}), 400

    judges_file = DATA_DIR / 'judges.json'
    if judges_file.exists():
        with open(judges_file, 'r') as f:
            judges = json.load(f)
    else:
        judges = []

    judges = [j for j in judges if j.get('name') != judge_name]

    with open(judges_file, 'w') as f:
        json.dump(judges, f, indent=2)

    return jsonify({'status': 'success', 'judges': judges})


@app.route('/run_evaluation', methods=['POST'])
def run_evaluation_endpoint():
    """Run evaluation using specified judge(s) - supports multiple judges"""


    data = request.json
    model_file = data.get('model_file')
    judges = data.get('judges')  # Can be a list or single judge dict
    task_id = data.get('task_id', f"eval_{id(data)}")

    # Handle both single judge (backwards compat) and multiple judges
    if data.get('judge'):
        judges = [data.get('judge')]
    elif not judges:
        return jsonify({'error': 'Missing judge or judges'}), 400

    # Ensure judges is a list
    if not isinstance(judges, list):
        judges = [judges]

    if not model_file:
        return jsonify({'error': 'Missing model_file'}), 400

    # Load weak model results
    model_path = DATA_DIR / model_file
    with open(model_path, 'r') as f:
        file_data = json.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(file_data, dict) and 'results' in file_data:
        metadata = file_data.get('metadata', {})
        results = file_data['results']
        strong_export = metadata.get('strong_export_file', 'unknown')
    else:
        # Old format - just a list
        results = file_data
        strong_export = 'unknown'

    # Extract model name from filename
    model_name = model_file.replace('weak_model_', '').replace('.json', '')

    # Create evaluation name with all judges
    judges_names = '_'.join([j['name'] for j in judges])
    eval_name = f"eval-{model_name}-{judges_names}"

    # Initialize progress tracking
    total_steps = len(results)
    progress_state[task_id] = {
        'current': 0,
        'total': total_steps,
        'message': f'Starting evaluation: {model_name} with {len(judges)} judge(s)...',
        'status': 'running'
    }

    # Create evaluation logger
    ev = weave.EvaluationLogger(
        name=eval_name,
        model=model_name
    )

    # Run evaluation
    for idx, example in enumerate(results):
        progress_state[task_id] = {
            'current': idx + 1,
            'total': total_steps,
            'message': f'{model_name} - Example {idx+1}/{total_steps}',
            'status': 'running'
        }
        # Skip examples with errors (null messages/output)
        if example.get('error') or not example.get('output'):
            continue

        strong_output = example.get('strong_model_output', '')
        weak_output = example.get('output', '')

        # Extract question
        question = ""
        messages = example.get('messages', [])
        if messages and len(messages) > 0:
            question = messages[0].get('content', '')

        # Run all judges and collect scores
        all_scores = {}
        for judge in judges:
            # Run judge
            if judge['type'] == 'llm':
                scores = run_llm_judge_eval(judge, strong_output, weak_output, question)
            else:
                scores = run_custom_judge_eval(judge, strong_output, weak_output)

            # Merge scores with judge name prefix to avoid conflicts
            for score_key, score_value in scores.items():
                all_scores[f"{judge['name']}_{score_key}"] = score_value

        # Log to weave with all scores from all judges
        ev.log_example(
            inputs={
                "question": question,
            },
            output={
                "strong_output": strong_output,
                "weak_output": weak_output

            },
            scores=all_scores
        )

    # Finish evaluation
    ev.log_summary()

    # Mark progress as complete
    progress_state[task_id] = {
        'current': total_steps,
        'total': total_steps,
        'message': 'Complete!',
        'status': 'complete'
    }

    return jsonify({
        'status': 'success',
        'evaluation_name': eval_name,
        'examples_evaluated': len(results),
        'weave_url': ev.ui_url,
        'strong_export': strong_export,
        'judges': [j['name'] for j in judges],
        'task_id': task_id
    })


def run_llm_judge_eval(judge, strong_output, weak_output, question):
    """Run LLM judge for evaluation endpoint using LLMAsAJudge framework"""

    # Build the prompt using user's template
    # Use replace instead of format to avoid issues with literal curly braces in prompts
    prompt = judge['prompt']
    if '{question}' in prompt:
        prompt = prompt.replace('{question}', question or '')
    if '{strong_output}' in prompt:
        prompt = prompt.replace('{strong_output}', strong_output or '')
    if '{weak_output}' in prompt:
        prompt = prompt.replace('{weak_output}', weak_output or '')

    # Custom parser to extract scores from response based on return type
    return_type = judge.get('returnType', '')

    def score_parser(response: str):
        """Parse the judge response based on return type"""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```'):
            # Remove ```json or ``` at start
            response = response.split('\n', 1)[1] if '\n' in response else response[3:]
            # Remove ``` at end
            if response.endswith('```'):
                response = response.rsplit('\n', 1)[0] if '\n' in response else response[:-3]
            response = response.strip()

        try:
            # Parse JSON response
            parsed = json.loads(response)

            if return_type == 'boolean':
                # Extract boolean value - return just the bool
                val = parsed.get('correct', parsed.get('result', parsed.get('value', False)))
                return bool(val)
            elif return_type == 'scalar':
                # Extract numeric score - return just the number
                val = parsed.get('score', parsed.get('scores', parsed.get('value', 0)))
                return float(val) if isinstance(val, (int, float)) else 0
            # elif return_type == 'string':
            #     # Extract string classification - return just the string
            #     val = parsed.get('classification', parsed.get('result', parsed.get('value', '')))
            #     return str(val)
            # else:
            #     # map or auto: return the entire dict
            #     return parsed if isinstance(parsed, dict) else {}
            else:
                # Unsupported return type
                print(f"Unsupported return type: {return_type}")
                return 0
        except:
            print(f"Failed to parse judge response as JSON: {response}")
            if return_type == 'scalar':
                return 0
            elif return_type == 'boolean':
                return False
            # elif return_type == 'string':
            #     return ''
            else:
                return 0

    try:
        # Initialize LLMAsAJudge with custom prompt
        judge_instance = LLMAsAJudge(
            models=[judge['model']],
            use_fully_custom_prompt=True,
            output_parser=score_parser,
            return_type=return_type if return_type else None
        )

        # Get judgment
        result = judge_instance.judge(prompt=prompt)

        # LLMAsAJudge wraps return values differently based on type:
        # - scalar: result['scores'] = the number, result['correct'] = the number
        # - boolean: result['correct'] = the bool, result['scores'] = None
        # - string: result['correct'] = the string, result['scores'] = None
        # - map: result['scores'] = the dict, result['correct'] = None

        if return_type == 'scalar':
            score_val = result.get('scores', result.get('correct', 0))
            return {'score': score_val}
        elif return_type == 'boolean':
            bool_val = result.get('correct', False)
            return {'correct': bool_val}
        # elif return_type == 'string':
        #     str_val = result.get('correct', '')
        #     return {'classification': str_val}
        # else:
        #     # map or auto
        #     return result.get('scores', {})
        else:
            # Unsupported return type - default to scalar
            score_val = result.get('scores', result.get('correct', 0))
            return {'score': score_val}

    except Exception as e:
        print(f"Error in LLM judge: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_custom_judge_eval(judge, strong_output, weak_output):
    """Run custom judge for evaluation endpoint"""
    try:
        namespace = {}
        exec(judge['customFunction'], namespace)
        judge_func = namespace.get('custom_judge')
        if judge_func:
            return judge_func(strong_output, weak_output)
    except Exception as e:
        print(f"Error in custom judge: {e}")
    return {}


@app.route('/fetch_traces', methods=['POST'])
def fetch_traces():
    """Fetch traces from a Weave project"""
    from quickdistill.get_traces import get_traces
    data = request.json
    project_name = data.get('project_name')

    if not project_name:
        return jsonify({'error': 'No project_name provided'}), 400

    try:
        # Call get_traces function directly
        output_file = get_traces(project_name)

        return jsonify({
            'status': 'success',
            'message': f'✓ Exported traces to {output_file}',
            'project_name': project_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/list_projects', methods=['GET'])
def list_projects():
    """List all projects with cached traces"""
    projects_dir = DATA_DIR / 'projects'
    if not projects_dir.exists():
        return jsonify({'projects': []})

    projects = []
    for project_path in projects_dir.iterdir():
        if project_path.is_dir():
            traces_file = project_path / 'traces_data.json'
            if traces_file.exists():
                # Get trace count and last modified time
                try:
                    with open(traces_file, 'r') as f:
                        traces = json.load(f)
                    count = len(traces)
                    last_modified = traces_file.stat().st_mtime
                except:
                    count = 0
                    last_modified = 0

                projects.append({
                    'name': project_path.name.replace('_', '/'),
                    'path': str(project_path),
                    'trace_count': count,
                    'last_modified': last_modified
                })

    return jsonify({'projects': projects})


@app.route('/get_preferences', methods=['GET'])
def get_preferences():
    """Get saved user preferences"""
    prefs_file = DATA_DIR / 'preferences.json'
    if prefs_file.exists():
        try:
            with open(prefs_file, 'r') as f:
                return jsonify(json.load(f))
        except:
            pass
    return jsonify({})


@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    """Save user preferences"""
    try:
        data = request.json
        prefs_file = DATA_DIR / 'preferences.json'
        with open(prefs_file, 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Routes for serving HTML pages
@app.route('/')
def index():
    """Serve the trace viewer page"""
    return send_from_directory(STATIC_DIR, 'trace_viewer.html')


@app.route('/judge')
def judge():
    """Serve the judge manager page"""
    return send_from_directory(STATIC_DIR, 'judge_manager.html')


@app.route('/projects/<path:filepath>')
def serve_project_file(filepath):
    """Serve files from the projects directory"""
    projects_dir = DATA_DIR / 'projects'

    # Security check - ensure the path doesn't escape the projects directory
    full_path = (projects_dir / filepath).resolve()
    if not str(full_path).startswith(str(projects_dir.resolve())):
        return jsonify({'error': 'Invalid path'}), 403

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(full_path.parent, full_path.name)


@app.route('/strong_exports/<path:filepath>')
def serve_strong_export(filepath):
    """Serve files from the strong_exports directory"""
    exports_dir = DATA_DIR / 'strong_exports'

    # Security check
    full_path = (exports_dir / filepath).resolve()
    if not str(full_path).startswith(str(exports_dir.resolve())):
        return jsonify({'error': 'Invalid path'}), 403

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(full_path.parent, full_path.name)


if __name__ == '__main__':
    print("Starting inference server on http://localhost:5001")
    print("Make sure WANDB_API_KEY is set in your environment")
    app.run(debug=True, port=5001)
