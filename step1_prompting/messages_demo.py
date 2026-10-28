from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"


def ask(messages, system=None):
    kwargs = {"system": system} if system else {}
    msg = client.messages.create(model=MODEL, max_tokens=100, messages=messages, **kwargs)
    return msg.content[0].text.strip(), msg.usage.input_tokens


# 1) No memory: two independent requests.
ask1, _ = ask([{"role": "user", "content": "My name is Alex."}])
ask2, tokens = ask([{"role": "user", "content": "What is my name?"}])
print("[1] Without history ->", ask2, f"(input tokens: {tokens})\n")

# 2) With memory: WE resend the whole history.
history = [
    {"role": "user", "content": "My name is Alex."},
    {"role": "assistant", "content": ask1},
    {"role": "user", "content": "What is my name?"},
]
answer, tokens = ask(history)
print("[2] With history    ->", answer, f"(input tokens: {tokens})\n")

# 3) System prompt changes behaviour.
question = [{"role": "user", "content": "What is LoRA?"}]
plain, _ = ask(question)
styled, _ = ask(question, system="You are a research assistant. Answer in exactly one sentence. If unsure, say you don't know.")
print("[3a] No system prompt:\n", plain, "\n")
print("[3b] With system prompt:\n", styled)
