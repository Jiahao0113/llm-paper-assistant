"""Generate a synthetic retrieval test set: one question per sampled chunk; that chunk is the gold answer location."""
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ConfigDict

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
PER_PAPER = 12
SEED = 0

PROMPT = """Below is a passage from a machine learning paper.
Write ONE question that a researcher might ask, which can be answered using ONLY this passage.
Requirements:
- The question must make sense on its own (no "this passage", "the paper above", "the authors" without naming the method).
- Do NOT copy phrases from the passage; paraphrase and use natural wording. Avoid the passage's rare terms when a general term would do.
- Ask about a specific fact, mechanism, or result, not something vague like "What does the paper say?".

Passage:
{text}"""


class Q(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


def usable(c) -> bool:
    t = c["text"]
    letters = sum(ch.isalpha() for ch in t)
    garbled = any(ord(ch) < 32 and ch not in "\n\t" for ch in t)
    return len(t) > 400 and letters / len(t) > 0.7 and not garbled and not re.match(r"^\s*(Table|Figure)\s+\d", t)


chunks = json.loads(Path("data/index/chunks.json").read_text())
rng = random.Random(SEED)
rows = []
for paper in sorted({c["paper"] for c in chunks}):
    pool = [i for i, c in enumerate(chunks) if c["paper"] == paper and usable(c)]
    for i in rng.sample(pool, PER_PAPER):
        msg = client.messages.create(
            model=MODEL, max_tokens=200, thinking={"type": "disabled"},
            messages=[{"role": "user", "content": PROMPT.format(text=chunks[i]["text"])}],
            output_config={"format": {"type": "json_schema", "schema": Q.model_json_schema()}},
        )
        q = Q.model_validate_json("".join(b.text for b in msg.content if b.type == "text")).question
        rows.append({"question": q, "gold_chunk": i, "paper": paper, "page": chunks[i]["page"]})

Path("eval/retrieval_set.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print(len(rows), "questions written")
