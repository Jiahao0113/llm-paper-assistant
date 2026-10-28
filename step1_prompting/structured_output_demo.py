import json
from typing import Literal

from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ConfigDict

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

QUESTION = "How does QLoRA differ from LoRA in memory usage, and which paper introduced it?"
PROMPT = f"Classify this question about papers and extract the method names it mentions.\n\nQuestion: {QUESTION}"


class QueryInfo(BaseModel):
    """The shape we want. Pydantic = a Python class that also validates data."""
    model_config = ConfigDict(extra="forbid")  # emits additionalProperties: false, which the API requires

    category: Literal["FACT", "COMPARE", "SUMMARY"]
    methods: list[str]
    needs_citation: bool


def text_of(msg):
    return "".join(b.text for b in msg.content if b.type == "text")


# A) Only ask nicely in the prompt.
msg = client.messages.create(
    model=MODEL, max_tokens=300, thinking={"type": "disabled"},
    messages=[{"role": "user", "content": PROMPT + "\n\nReply with JSON with keys category, methods, needs_citation."}],
)
raw = text_of(msg)
print("[A] raw output:\n", raw, "\n")
try:
    print("[A] json.loads OK:", json.loads(raw))
except json.JSONDecodeError as e:
    print("[A] json.loads FAILED:", e)

# B) Schema-constrained output: the API guarantees the output matches the JSON schema.
msg = client.messages.create(
    model=MODEL, max_tokens=300, thinking={"type": "disabled"},
    messages=[{"role": "user", "content": PROMPT}],
    output_config={"format": {"type": "json_schema", "schema": QueryInfo.model_json_schema()}},
)
raw = text_of(msg)
print("\n[B] raw output:\n", raw)
info = QueryInfo.model_validate_json(raw)  # parse + validate into a typed object
print("[B] typed object:", info)
print("[B] info.category == 'COMPARE' ->", info.category == "COMPARE")
