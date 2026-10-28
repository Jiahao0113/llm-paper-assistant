import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompt = "The cat sat on the"
input_ids = tok(prompt, return_tensors="pt").input_ids  # shape (1, seq_len)

# One forward pass: the model returns logits for EVERY position; we only want the last one,
# which predicts the token that comes after the prompt.
with torch.no_grad():
    logits = model(input_ids).logits[0, -1]  # shape (vocab_size,)
print("logits shape:", tuple(logits.shape), "(one raw score per vocab token)\n")


def top_tokens(logits, temperature=1.0, k=5):
    probs = torch.softmax(logits / temperature, dim=-1)
    top = torch.topk(probs, k)
    return [(tok.decode(i), round(p.item(), 3)) for p, i in zip(top.values, top.indices)]


for T in [0.5, 1.0, 2.0]:
    print(f"T={T}: {top_tokens(logits, T)}")


def top_p_filter(probs, p):
    """Keep the smallest set of tokens whose cumulative probability >= p, renormalize."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # keep a token if the cumulative prob BEFORE it is still < p (so the token that crosses p is kept)
    keep = (cum - sorted_probs) < p
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum(), int(keep.sum())


probs = torch.softmax(logits, dim=-1)
print()
for p in [1.0, 0.9, 0.5, 0.1]:
    _, n = top_p_filter(probs, p)
    print(f"top_p={p}: keeps {n} of {len(probs)} tokens")


def generate(prompt, steps=15, temperature=1.0, top_p=1.0, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = tok(prompt, return_tensors="pt").input_ids
    for _ in range(steps):  # autoregressive loop
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        if temperature == 0:  # greedy
            nxt = torch.argmax(logits)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs, _ = top_p_filter(probs, top_p)
            nxt = torch.multinomial(probs, 1, generator=g)[0]
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return tok.decode(ids[0])


print("\n--- generation ---")
for label, kw in [("greedy (T=0)", dict(temperature=0)),
                  ("T=1.0", dict(temperature=1.0)),
                  ("T=2.0", dict(temperature=2.0)),
                  ("T=1.0, top_p=0.9", dict(temperature=1.0, top_p=0.9))]:
    print(f"\n{label}")
    for seed in range(3):
        print("  ", repr(generate("The cat sat on the", seed=seed, **kw)))
