import weave
import json
import sys
from pathlib import Path

def serialize_value(obj):
    """Convert Weave objects and other non-serializable types to JSON-safe format"""
    if obj is None:
        return None

    # Handle basic JSON-serializable types FIRST
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists/tuples BEFORE checking for weave objects
    if isinstance(obj, (list, tuple)):
        return [serialize_value(item) for item in obj]

    # Handle dictionaries BEFORE checking for weave objects
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}

    # Handle WeaveObject and ObjectRef types - convert to string only if not handled above
    if hasattr(obj, '__class__') and 'weave' in obj.__class__.__module__.lower():
        return str(obj)

    # Fallback to string representation
    return str(obj)


def get_traces(project_name):
    """Fetch traces from Weave project and save to projects/{project_name}/traces_data.json"""

    print(f"Fetching traces from project: {project_name}")

    # Initialize weave client
    client = weave.init(project_name)

    # Get all calls, no time filter
    calls = client.get_calls(
        sort_by=[{"field": "started_at", "direction": "desc"}]
    )

    traces_data = []

    for call in calls:
        # Convert to dict and serialize
        inputs = dict(call.inputs) if hasattr(call.inputs, 'keys') else call.inputs
        output = dict(call.output) if hasattr(call.output, 'keys') else call.output

        # Serialize inputs and outputs to handle Weave objects
        inputs = serialize_value(inputs)
        output = serialize_value(output)

        # Extract op_name (function name)
        op_name = call.op_name if call.op_name else 'unknown'
        # Get just the function name part (after the last /)
        op_display_name = op_name.split('/')[-1].split(':')[0] if '/' in op_name else op_name

        # Determine model
        models = []
        if call.summary and "usage" in call.summary:
            models = list(call.summary["usage"].keys())
        model = models[0] if models else (inputs.get('model', 'N/A') if isinstance(inputs, dict) else 'N/A')

        # Extract messages (for OpenAI calls) - check in inputs
        messages = []
        if isinstance(inputs, dict):
            # Messages could be at top level or nested in inputs
            messages = inputs.get('messages', [])
            # If messages is not a list (could be a string), try to handle it
            if not isinstance(messages, list):
                messages = []

        # Extract output content
        output_content = None
        if isinstance(output, dict):
            # For OpenAI API calls
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'message' in choice:
                    output_content = choice['message'].get('content', '')
            else:
                # For other calls, just stringify the output
                try:
                    output_content = json.dumps(output, indent=2)
                except:
                    output_content = str(output)
        else:
            output_content = str(output)

        trace_entry = {
            'id': call.id,
            'op_name': op_name,
            'op_display_name': op_display_name,
            'model': model,
            'time': call.started_at.isoformat() if call.started_at else None,
            'inputs': inputs,
            'messages': messages,
            'output': output_content,
            'temperature': inputs.get('temperature') if isinstance(inputs, dict) else None,
            'max_tokens': inputs.get('max_tokens') if isinstance(inputs, dict) else None,
            'usage': call.summary.get('usage', {}).get(model, {}) if call.summary and 'usage' in call.summary else {}
        }

        traces_data.append(trace_entry)

    # Create projects directory if it doesn't exist
    projects_dir = Path('projects')
    projects_dir.mkdir(exist_ok=True)

    # Create project-specific directory
    project_dir = projects_dir / project_name.replace('/', '_')
    project_dir.mkdir(exist_ok=True)

    # Save to JSON file in project directory
    output_file = project_dir / 'traces_data.json'
    with open(output_file, 'w') as f:
        json.dump(traces_data, f, indent=2)

    print(f"✓ Exported {len(traces_data)} traces to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    # Get project name from command line argument
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "byyoung3/arena-detailed"

    get_traces(project_name)
