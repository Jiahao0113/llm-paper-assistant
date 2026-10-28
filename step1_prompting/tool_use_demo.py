import json

from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

# A fake "database" standing in for real retrieval (we build the real one in Step 2-3).
PAPERS = {
    "LoRA": "LoRA (Hu et al., 2021) freezes the pretrained weights and trains low-rank matrices A and B injected into attention layers.",
    "QLoRA": "QLoRA (Dettmers et al., 2023) fine-tunes LoRA adapters on top of a 4-bit quantized base model, cutting memory a lot.",
}


def search_papers(query: str) -> str:
    """The real Python function. The model never runs this; our code does."""
    hits = [text for name, text in PAPERS.items() if name.lower() in query.lower()]
    return "\n".join(hits) if hits else "No results."


# Tool description sent to the model: name + what it does + JSON schema of its arguments.
tools = [{
    "name": "search_papers",
    "description": "Search the paper database. Query should contain a method name such as LoRA or QLoRA.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Method name to look up"}},
        "required": ["query"],
    },
}]

messages = [{"role": "user", "content": "How does QLoRA reduce memory compared with LoRA?"}]


def call_model():
    return client.messages.create(
        model=MODEL, max_tokens=500, tools=tools, messages=messages, thinking={"type": "disabled"}
    )


# Step 1: the model sees the question and the tool list.
response = call_model()
print("stop_reason:", response.stop_reason)

# The agent loop in miniature: keep going while the model asks for tools.
while response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})  # keep the model's request in history

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"  model requests: {block.name}({json.dumps(block.input)})")
            result = search_papers(**block.input)  # OUR code executes the tool
            print(f"  we return: {result[:70]}...")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

    messages.append({"role": "user", "content": tool_results})  # results go back as a user message
    response = call_model()
    print("stop_reason:", response.stop_reason)

print("\nFinal answer:\n", "".join(b.text for b in response.content if b.type == "text"))
