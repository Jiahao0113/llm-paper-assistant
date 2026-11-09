"""Build a human audit of the LLM judge: a balanced sample of factual sentences with the passages they cite, WITHOUT the judge's verdict.
Label each item S (the cited passages support the claim) or U (they do not), then run eval/audit_score.py.
Usage: python -m eval.audit_sheet [n_per_class]   (default 15 supported + 15 unsupported, spread over systems)"""
import csv
import json
import random
import sys
from pathlib import Path

from src.copilot.retrieval import DenseIndex

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rnd = random.Random(7)
chunks = DenseIndex()
chunks.load()
items = []
for x in json.loads(Path("eval/research_results.json").read_text()):
    for v in x.get("sentence_verdicts", []):
        if v["cited"]:  # sentences with no citation are unsupported by definition; the interesting question is the judged ones
            items.append({"system": x["system"], "task": x["task"], "run": x["run"], **v})
sample = []
for label in (True, False):
    pool = [i for i in items if i["supported"] == label]
    rnd.shuffle(pool)
    sample += pool[:N]
rnd.shuffle(sample)

out = Path("eval/audit")
out.mkdir(exist_ok=True)
lines = ["# Judge audit\n", "For each item, read the sentence and the passages it cites. Write S if the passages state or clearly imply the claim, U if they do not.",
         "Put your labels in `labels.csv` (column `human`). Do not look at `key.json`; it holds the judge's verdicts.",
         "Items are shuffled, so any run of items from the top is roughly balanced: label from Item 1 downward and stop whenever you like (about 20 gives a usable result; unlabelled rows are skipped).\n"]
for n, it in enumerate(sample, 1):
    lines.append(f"## Item {n}\n**Sentence:** {it['text']}\n")
    for c in it["cited"]:
        ch = chunks.chunks[c]
        lines.append(f"- **[c{c}]** ({ch.paper}, p{ch.page}) {ch.text}\n")
(out / "audit_sheet.md").write_text("\n".join(lines))
(out / "key.json").write_text(json.dumps([{"item": n, "judge_supported": it["supported"], "system": it["system"], "task": it["task"], "run": it["run"]} for n, it in enumerate(sample, 1)], indent=1))
with open(out / "labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "human"])
    w.writerows([[n, ""] for n in range(1, len(sample) + 1)])
print(f"wrote {len(sample)} items to {out}/ (sheet, labels.csv to fill in, key.json)")
