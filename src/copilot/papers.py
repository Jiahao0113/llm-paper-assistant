"""The paper library: data/papers.json is the manifest (arXiv id + title); PDFs live in data/raw/ (gitignored)."""
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

MANIFEST = Path("data/papers.json")
RAW = Path("data/raw")
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def load_papers() -> dict[str, str]:
    """id -> title, in manifest order (the original five papers first, so their chunk ids stay stable)."""
    return {p["id"]: p["title"] for p in json.loads(MANIFEST.read_text())}


def fetch_titles(ids: list[str]) -> dict[str, str]:
    url = "http://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    out = {}
    for e in root.findall("a:entry", ATOM):
        arxiv_id = re.sub(r"v\d+$", "", e.find("a:id", ATOM).text.rsplit("/", 1)[-1])
        out[arxiv_id] = " ".join(e.find("a:title", ATOM).text.split())
    return out


def download(ids, delay=3.0):
    """Polite download (arXiv asks for ~1 request per 3 s). Skips files that already exist."""
    RAW.mkdir(parents=True, exist_ok=True)
    for i in ids:
        dest = RAW / f"{i}.pdf"
        if dest.exists():
            continue
        data = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{i}", headers={"User-Agent": "llm-paper-assistant"}), timeout=120).read()
        assert data[:4] == b"%PDF", f"{i}: not a PDF"
        dest.write_bytes(data)
        print("downloaded", i, f"{len(data)/1e6:.1f} MB", flush=True)
        time.sleep(delay)
