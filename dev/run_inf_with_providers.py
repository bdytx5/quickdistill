import os
import anthropic

# ---------------- GEMINI ----------------
from google import genai
# ---------------- GROK ----------------
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user, system

import weave; weave.init("providers-testing")


def run_gemini(prompt: str):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    )

    return resp.text


# ---------------- CLAUDE ----------------

def run_claude(prompt: str):
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    msg = client.messages.create(
        model="claude-4.5-haiku",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return msg.content[0].text



def run_grok(prompt: str):
    client = XAIClient(
        api_key=os.environ["XAI_API_KEY"],
        timeout=3600
    )

    chat = client.chat.create(
        model="grok-4-1-fast-reasoning"
    )

    chat.append(system("You are Grok, a helpful AI assistant."))
    chat.append(user(prompt))

    resp = chat.sample()

    return resp.content


# ---------------- UNIFIED ROUTER ----------------
def run_model(provider: str, prompt: str):
    provider = provider.lower()

    if provider == "gemini":
        return run_gemini(prompt)

    if provider == "claude":
        return run_claude(prompt)

    if provider == "grok":
        return run_grok(prompt)

    raise ValueError(provider)


# ---------------- TEST ----------------
if __name__ == "__main__":
    prompt = "Explain transformers simply"

    for provider in ["gemini", "claude", "grok"]:
        try:
            print(f"\n=== {provider.upper()} ===")
            print(run_model(provider, prompt))
        except Exception as e:
            print(provider, "failed:", e)