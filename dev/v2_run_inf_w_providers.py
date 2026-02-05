# pip install openai anthropic google-genai

import os

# ================= OPENAI =================
from openai import OpenAI
from google import genai
import anthropic

import weave; weave.init("providers-testing")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])








def openai_responses(prompt: str):
    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return resp.output_text


def openai_chat(prompt: str):
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def openai_stream(prompt: str):
    print("\n[OpenAI Streaming]")
    with openai_client.responses.stream(
        model="gpt-5-mini",
        input=prompt
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
    print()


# ================= ANTHROPIC =================


anthropic_client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)


def anthropic_messages(prompt: str):
    resp = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text


def anthropic_stream(prompt: str):
    print("\n[Anthropic Streaming]")
    with anthropic_client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


# ================= GEMINI =================

gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


def gemini_generate(prompt: str):
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    )
    return resp.text


def gemini_chat(prompt: str):
    chat = gemini_client.chats.create(
        model="gemini-2.5-flash"
    )
    resp = chat.send_message(prompt)
    return resp.text


def gemini_stream(prompt: str):
    print("\n[Gemini Streaming]")
    chat = gemini_client.chats.create(
        model="gemini-2.5-flash"
    )
    stream = chat.send_message_stream(prompt)

    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


# ================= TOOL CALL EXAMPLE =================
# Minimal cross-provider demonstration using OpenAI only
# (Anthropic/Gemini support tools but schemas differ heavily)

def openai_tool_example():

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]

    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "What's weather in Tokyo?"}],
        tools=tools
    )

    return resp.choices[0].message.tool_calls


# ================= RUN ALL =================

if __name__ == "__main__":

    prompt = "Explain transformers simply."

    # print("\n==== OPENAI RESPONSES ====")
    # print(openai_responses(prompt))

    # print("\n==== OPENAI CHAT ====")
    # print(openai_chat(prompt))

    # openai_stream(prompt)

    print("\n==== ANTHROPIC ====")
    print(anthropic_messages(prompt))

    anthropic_stream(prompt)

    print("\n==== GEMINI GENERATE ====")
    print(gemini_generate(prompt))

    print("\n==== GEMINI CHAT ====")
    print(gemini_chat(prompt))

    gemini_stream(prompt)

    print("\n==== OPENAI TOOL CALL ====")
    print(openai_tool_example())