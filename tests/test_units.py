import json
from pathlib import Path

import numpy as np

from src.copilot.bm25 import BM25, rrf
from src.copilot.research import cited_ids, split_sentences

ROOT = Path(__file__).resolve().parent.parent


def test_cited_ids_handles_adjacent_and_grouped_brackets():
    assert cited_ids("Claim one [c12][c40]. Claim two [c12, c13].") == [12, 40, 13]
    assert cited_ids("No citation here.") == []
    assert cited_ids("Ordinary brackets [1] and [see above] are ignored.") == []


def test_split_sentences_drops_headings_and_fragments():
    text = ("# Title\n\nLoRA freezes the base weights and trains small matrices [c1]. "
            "It adds no extra inference latency after merging [c2].\nOK.")
    sents = split_sentences(text)
    assert len(sents) == 2 and all("Title" not in s for s in sents)


def test_split_sentences_ignores_very_short_sentences():
    """Known limitation: a factual sentence under 26 characters is not checked by the verifier or the judge."""
    assert split_sentences("It adds no latency [c2].") == []


def test_rrf_prefers_items_ranked_high_by_both_lists():
    fused = rrf([[3, 1, 2], [1, 3, 2]])
    assert set(fused[:2]) == {1, 3} and fused[-1] == 2


def test_bm25_ranks_exact_term_match_first():
    docs = ["attention is all you need", "low rank adaptation of language models", "retrieval augmented generation"]
    scores = BM25(docs).scores("low rank adaptation")
    assert int(np.argmax(scores)) == 1


def test_manifest_keeps_original_five_papers_first_and_sorted():
    """Gold labels in the older eval sets are chunk indices; they stay valid only while these five come first, in this order."""
    ids = [p["id"] for p in json.loads((ROOT / "data/papers.json").read_text())]
    assert ids[:5] == ["1706.03762", "2005.11401", "2106.09685", "2210.03629", "2305.14314"]
    assert len(ids) == len(set(ids))


def test_research_tasks_reference_papers_in_the_manifest():
    manifest = {p["id"] for p in json.loads((ROOT / "data/papers.json").read_text())}
    for t in json.loads((ROOT / "eval/research_tasks.json").read_text())["tasks"]:
        assert set(t["required_papers"]) <= manifest, t["id"]
        assert {k["paper"] for k in t["key_points"]} <= manifest, t["id"]
