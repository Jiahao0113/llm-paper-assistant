from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

# Task: label a user question about papers as FACT, COMPARE or SUMMARY (a simple "query router").
TASK = "Classify the question as FACT, COMPARE or SUMMARY."
test_questions = [
    "What learning rate did the LoRA paper use?",
    "How does QLoRA differ from LoRA in memory usage?",
    "Give me an overview of the Transformer paper.",
]


def run(messages):
    msg = client.messages.create(
        model=MODEL, max_tokens=100, messages=messages, thinking={"type": "disabled"}
    )
    # The response is a list of content blocks; keep only the text ones.
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text.strip().replace("\n", " ")


print("=== zero-shot: instruction only ===")
for q in test_questions:
    print(f"{q!r} -> {run([{'role': 'user', 'content': f'{TASK}\n\nQuestion: {q}'}])}")

# Few-shot: show examples as previous turns of the conversation, then the real question.
examples = [
    ("Which dataset was BERT pretrained on?", "FACT"),
    ("Compare GPT-3 and T5 in terms of architecture.", "COMPARE"),
    ("Summarize the main ideas of the RLHF paper.", "SUMMARY"),
]
shots = []
for q, label in examples:
    shots += [{"role": "user", "content": f"{TASK}\n\nQuestion: {q}"},
              {"role": "assistant", "content": label}]

print("\n=== few-shot: 3 examples ===")
for q in test_questions:
    msgs = shots + [{"role": "user", "content": f"{TASK}\n\nQuestion: {q}"}]
    print(f"{q!r} -> {run(msgs)}")
