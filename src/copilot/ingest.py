"""PDF -> clean paragraphs -> chunks with metadata. Baseline version: tables/figures/formulas are not handled specially."""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # used only to measure chunk size in tokens

MAX_TOKENS = 300
OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    paper: str    # file stem, e.g. "2106.09685"
    title: str
    section: str
    page: int     # page where the chunk starts (1-based)
    text: str


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)    # ligatures: "ﬁ" -> "fi"
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # re-join words hyphenated at a line break
    text = re.sub(r"(?<=[A-Z])- (?=[A-Z])", "", text)  # "LAN- GUAGE" -> "LANGUAGE" (hyphenated uppercase titles)
    return re.sub(r"\s+", " ", text).strip()


def is_heading(text: str) -> bool:
    short = len(text) < 80
    numbered = re.match(r"^\d+(\.\d+)*\s+\S", text) is not None
    shouty = text.isupper() and sum(c.isalpha() for c in text) >= 4  # excludes figure labels like "B= 0", "Θ"
    return short and (numbered or shouty)


def is_table_residue(raw: str) -> bool:
    """Tables extract as many very short lines (numbers, labels). Prose has long lines."""
    lines = [l for l in raw.split("\n") if l.strip()]
    return len(lines) >= 5 and sum(len(l) for l in lines) / len(lines) < 14


def load_paragraphs(pdf_path: Path):
    """Return a list of (page_number, raw_block_text) for the whole paper, in reading order."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        pages.append([(pno, b[4].strip()) for b in blocks if b[6] == 0 and b[4].strip()])  # b[6]==0: text block

    # Headers/footers: a short block whose text repeats on many pages.
    counts = Counter(t for blocks in pages for _, t in blocks if len(t) < 80)
    repeated = {t for t, c in counts.items() if c >= max(3, len(pages) // 3)}

    out = []
    for blocks in pages:
        for pno, t in blocks:
            if t in repeated or re.fullmatch(r"\d{1,3}", t):  # header/footer or bare page number
                continue
            out.append((pno, t))
    return out


def chunk_paper(pdf_path: Path) -> list[Chunk]:
    paper = pdf_path.stem
    paras = load_paragraphs(pdf_path)
    title = clean(paras[0][1]) if paras else paper
    chunks: list[Chunk] = []
    section, buf, buf_page = "Front matter", [], 1

    def flush():
        nonlocal buf
        if buf:
            chunks.append(Chunk(paper, title, section, buf_page, " ".join(buf)))
            # keep the tail of the finished chunk as overlap for the next one
            tail = []
            for p in reversed(buf):
                if n_tokens(" ".join(tail + [p])) > OVERLAP_TOKENS:
                    break
                tail.insert(0, p)
            buf = tail

    for pno, raw in paras:
        text = clean(raw)
        if re.fullmatch(r"(references|bibliography)", text, re.I):
            break  # v1: drop references AND everything after (appendices are lost too)
        if is_heading(text):
            flush()
            buf = []  # no overlap across sections
            section = text
            continue
        if is_table_residue(raw) or len(text) < 40:
            continue
        if not buf:
            buf_page = pno
        if buf and n_tokens(" ".join(buf + [text])) > MAX_TOKENS:
            flush()
            buf_page = pno
        buf.append(text)
    flush()
    return chunks


if __name__ == "__main__":
    import statistics

    for pdf in sorted(Path("data/raw").glob("*.pdf")):
        cs = chunk_paper(pdf)
        sizes = [n_tokens(c.text) for c in cs]
        print(f"{pdf.stem}: {len(cs)} chunks, tokens min/median/max = {min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}")
