"""Compare an audit labeller with the judge's verdicts: agreement, Cohen's kappa, and the disagreements.
Usage: python -m eval.audit_score [labels.csv] [column]   (default: eval/audit/labels.csv, column `human`).
The labeller is named in the output so a model audit is never mistaken for a human one."""
import csv
import json
import sys
from pathlib import Path

key = {k["item"]: k for k in json.loads(Path("eval/audit/key.json").read_text())}
path = sys.argv[1] if len(sys.argv) > 1 else "eval/audit/labels.csv"
column = sys.argv[2] if len(sys.argv) > 2 else "human"
labels = {int(r["item"]): r[column].strip().upper() for r in csv.DictReader(open(path)) if r[column].strip()}
assert labels, "no labels filled in"
pairs = [(labels[i] == "S", key[i]["judge_supported"], i) for i in labels]
n = len(pairs)
agree = sum(h == j for h, j, _ in pairs)
tp = sum(h and j for h, j, _ in pairs); tn = sum((not h) and (not j) for h, j, _ in pairs)
fp = sum((not h) and j for h, j, _ in pairs)  # judge says supported, human says not (judge too lenient)
fn = sum(h and (not j) for h, j, _ in pairs)  # judge says unsupported, human says supported (judge too strict)
pe = ((tp + fn) / n) * ((tp + fp) / n) + ((tn + fp) / n) * ((tn + fn) / n)
kappa = (agree / n - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"labeller: {column} ({path})\nitems labelled: {n}\nagreement: {agree}/{n} = {agree / n:.1%}\nCohen's kappa: {kappa:.2f}")
print(f"judge too lenient (judge S, {column} U): {fp}\njudge too strict (judge U, {column} S): {fn}")
print("disagreeing items:", [i for h, j, i in pairs if h != j])
