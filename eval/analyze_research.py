"""Offline analysis of eval/research_results.json (no API calls).
Comparisons are paired on (task, run). Because the same tasks are re-run, runs are not independent samples, so the confidence
intervals resample TASKS (bootstrap), averaging each task's paired differences over its runs first."""
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["single_shot", "react", "staged_noverify", "staged_r1", "staged", "graph"]
results = [x for x in json.loads(Path("eval/research_results.json").read_text()) if "error" not in x]
by = defaultdict(dict)  # (task, run) -> system -> result
for x in results:
    by[(x["task"], x["run"])][x["system"]] = x


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def table():
    print(f"{'system':16s}{'reports':>8s}{'runs':>6s}{'support':>9s}{'uncited':>9s}{'keypts':>8s}{'papers':>8s}{'calls':>7s}{'$/task':>9s}{'sec':>6s}")
    for s in ORDER:
        R = [x for x in results if x["system"] == s and x["type"] != "gap"]
        if R:
            runs = sorted({x["run"] for x in R})
            print(f"{s:16s}{len(R):8d}{len(runs):6d}{mean([x['support'] for x in R]):9.3f}{mean([x['uncited'] for x in R]):9.2f}"
                  f"{mean([x.get('key_points') for x in R]):8.3f}{mean([x.get('papers') for x in R]):8.3f}{mean([x['llm_calls'] for x in R]):7.1f}"
                  f"{mean([x['cost'] for x in R]):9.4f}{mean([x['seconds'] for x in R]):6.0f}")
    print("\nper-run support (shows run-to-run spread):")
    for s in ORDER:
        R = [x for x in results if x["system"] == s and x["type"] != "gap"]
        if R:
            print(f"  {s:16s}", "  ".join(f"run{k}: {mean([x['support'] for x in R if x['run'] == k]):.3f}" for k in sorted({x['run'] for x in R})))


def task_diffs(a, b, key):
    """Per-task mean over runs of (a - b), using only runs where both systems have a result for that task."""
    per_task = defaultdict(list)
    for (task, _), m in by.items():
        if a in m and b in m and m[a]["type"] != "gap" and m[a].get(key) is not None and m[b].get(key) is not None:
            per_task[task].append(m[a][key] - m[b][key])
    return {t: st.mean(v) for t, v in per_task.items()}


def bootstrap_ci(values, n=10000, seed=0):
    rnd = random.Random(seed)
    means = sorted(st.mean(rnd.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def comparisons():
    print("\npaired differences (a - b) by task, 95% bootstrap CI over tasks; a CI that excludes 0 is evidence of a real difference:")
    print(f"  {'a':16s} {'b':16s} {'metric':11s}{'tasks':>6s}{'mean':>9s}   {'95% CI':>17s}   tasks a better / tied / worse")
    for a, b in [("react", "single_shot"), ("staged_noverify", "react"), ("staged", "staged_noverify"), ("staged_r1", "staged_noverify"), ("staged", "staged_r1"), ("staged", "react"), ("graph", "staged")]:
        for key in ("support", "key_points", "uncited"):
            d = task_diffs(a, b, key)
            if len(d) >= 5:
                v = list(d.values())
                lo, hi = bootstrap_ci(v)
                better = (lambda x: x > 1e-9) if key != "uncited" else (lambda x: x < -1e-9)
                worse = (lambda x: x < -1e-9) if key != "uncited" else (lambda x: x > 1e-9)
                print(f"  {a:16s} {b:16s} {key:11s}{len(v):6d}{st.mean(v):+9.3f}   [{lo:+.3f}, {hi:+.3f}]   {sum(map(better, v)):2d} / {sum(not better(x) and not worse(x) for x in v):2d} / {sum(map(worse, v)):2d}")


def decomposition():
    """`support` mixes two things: how many factual sentences carry a citation, and how often a cited sentence is actually supported.
    Only reports that stored per-sentence verdicts can be split (the later runs), so the subsets differ in size between systems."""
    print("\nsupport split into its two parts (reports with per-sentence verdicts only; subsets are not matched across systems):")
    print(f"  {'system':16s}{'reports':>8s}{'factual':>9s}{'cited share':>13s}{'cited sentences supported':>27s}")
    for s in ORDER:
        reps = [x for x in results if x["system"] == s and x["type"] != "gap" and "sentence_verdicts" in x]
        vs = [v for x in reps for v in x["sentence_verdicts"]]
        cited = [v for v in vs if v["cited"]]
        if cited:
            print(f"  {s:16s}{len(reps):8d}{len(vs):9d}{len(cited) / len(vs):13.1%}{sum(v['supported'] for v in cited) / len(cited):27.1%}")


def verifier_behaviour():
    print("\nverifier behaviour (the pipeline's own Haiku verifier, from the stored traces; not the Sonnet judge):")
    for s in ("staged_r1", "staged", "graph"):
        R = [x for x in results if x["system"] == s and x["type"] != "gap"]
        first, last, repaired = [], [], 0
        for x in R:
            v = [t for t in x["trace"] if t["stage"] == "verify"]
            if v:
                first.append(v[0]["unsupported"] / max(v[0]["sentences"], 1))
                last.append(v[-1]["unsupported"] / max(v[-1]["sentences"], 1))
                repaired += any(t["stage"] == "repair" for t in x["trace"])
        if first:
            print(f"  {s:8s} n={len(first)}: flagged on first check {mean(first):.1%} of sentences -> {mean(last):.1%} after repair; "
                  f"{repaired}/{len(first)} briefs needed a repair; {sum(l == 0 for l in last)}/{len(last)} ended with nothing flagged")


def gaps():
    print("\ngap tasks (brief said what is missing / invented details about it):")
    for s in ORDER:
        G = [x for x in results if x["system"] == s and x["type"] == "gap"]
        if G:
            print(f"  {s:16s} acknowledged {sum(x['acknowledged_gap'] for x in G)}/{len(G)}, fabricated {sum(x['fabricated'] for x in G)}/{len(G)}")


if __name__ == "__main__":
    table()
    comparisons()
    decomposition()
    verifier_behaviour()
    gaps()
