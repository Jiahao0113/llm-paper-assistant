from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say hi in 5 words."}],
)
print(msg.content[0].text)
print("tokens:", msg.usage.input_tokens, "in /", msg.usage.output_tokens, "out")
