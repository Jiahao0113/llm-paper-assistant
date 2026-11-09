"""Offline analysis of eval/research_results.json (no API calls). Uses runs 0 and 1 only; later runs were cut short by an API usage limit.
Paired comparisons match the same task and run index, and count wins/ties/losses so one outlier cannot drive the mean."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["single_shot", "react", "staged_noverify", "staged", "graph"]
results = [x for x in json.loads(Path("eval/research_results.json").read_text()) if "error" not in x and x["run"] <= 1]
by = defaultdict(dict)
for x in results:
    by[(x["task"], x["run"])][x["system"]] = x


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


print(f"{'system':16s}{'reports':>8s}{'support':>9s}{'uncited':>9s}{'keypts':>8s}{'papers':>8s}{'calls':>7s}{'$/task':>9s}{'sec':>6s}")
for s in ORDER:
    R = [x for x in results if x["system"] == s and x["type"] != "gap"]
    if R:
        print(f"{s:16s}{len(R):8d}{mean([x['support'] for x in R]):9.3f}{mean([x['uncited'] for x in R]):9.2f}{mean([x.get('key_points') for x in R]):8.3f}"
              f"{mean([x.get('papers') for x in R]):8.3f}{mean([x['llm_calls'] for x in R]):7.1f}{mean([x['cost'] for x in R]):9.4f}{mean([x['seconds'] for x in R]):6.0f}")

print("\npaired differences (a - b), same task and run:")
for a, b in [("react", "single_shot"), ("staged_noverify", "react"), ("staged", "staged_noverify"), ("staged", "react"), ("graph", "staged")]:
    for key in ("support", "key_points", "uncited"):
        d = [m[a][key] - m[b][key] for m in by.values() if a in m and b in m and m[a]["type"] != "gap" and m[a].get(key) is not None and m[b].get(key) is not None]
        if d:
            print(f"  {a:16s} - {b:16s} {key:10s} n={len(d):2d} mean {st.mean(d):+.3f}   wins {sum(x > 0 for x in d):2d} ties {sum(x == 0 for x in d):2d} losses {sum(x < 0 for x in d):2d}")

print("\ngap tasks (brief said what is missing / invented details about it):")
for s in ORDER:
    G = [x for x in results if x["system"] == s and x["type"] == "gap"]
    if G:
        print(f"  {s:16s} acknowledged {sum(x['acknowledged_gap'] for x in G)}/{len(G)}, fabricated {sum(x['fabricated'] for x in G)}/{len(G)}")
