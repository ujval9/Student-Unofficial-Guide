"""
ingest.py — Stage 1 + 2 of the RAG pipeline: Document Ingestion -> Chunking.

Domain: off-campus housing for University at Buffalo students (see planning.md).

What it does:
  1. INGEST  — for each source in SOURCES, load the raw page. URLs are fetched
               over HTTP; any source you saved by hand into documents/ is loaded
               from disk instead (needed for sites that block scraping, e.g.
               Reddit and apartments.com). Raw text is saved to raw/raw_documents.jsonl
               BEFORE any cleaning, so cleaning is reproducible and re-runnable.
  2. CLEAN   — strip HTML tags, nav menus, cookie banners, ads, footers, repeated
               headers, share/"read more" links, comment counts, and HTML entities.
  3. CHUNK   — split each cleaned document into token-based, overlapping chunks
               using a tokenizer compatible with the embedding model.

Chunking spec (planning.md -> Chunking Strategy):
    chunk size = 800-1,000 tokens   -> CHUNK_TOKENS  = 900  (midpoint)
    overlap    = 150-200 tokens     -> OVERLAP_TOKENS = 175 (midpoint)

Usage:
    .venv/bin/python ingest.py
"""

import html
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

# --- Configuration (from planning.md) ----------------------------------------
DOCUMENTS_DIR = Path("documents")            # manually-saved pages (.html/.txt)
RAW_DIR = Path("raw")                        # raw text, saved before cleaning
RAW_FILE = RAW_DIR / "raw_documents.jsonl"
CHUNKS_FILE = Path("chunks.json")

# Revised down from the planned 800-1,000: the real corpus turned out far
# shorter than expected (~9k tokens total), so large chunks gave 1 chunk per
# document and only ~16 chunks total (under the 50 floor). 250/50 keeps the
# same ~20% overlap ratio while isolating individual recommendations. See
# planning.md -> Chunking Strategy for the full rationale.
CHUNK_TOKENS = 250                           # target tokens per chunk
OVERLAP_TOKENS = 50                          # token overlap between chunks

# nomic-embed-text-v1.5 (the planned embedding model) is built on the
# bert-base-uncased WordPiece tokenizer, so token counts here match what the
# embedding model will see.
TOKENIZER_NAME = "bert-base-uncased"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (UB housing RAG student project)"}
HTTP_TIMEOUT = 25

# The 10 sources from planning.md. `slug` is the local filename stem: to ingest
# a blocked source, save the page as documents/<slug>.html (or .txt).
SOURCES = [
    {"slug": "01_reddit_best_off_campus",      "title": "r/UBreddit — best off campus housing",            "url": "https://www.reddit.com/r/UBreddit/comments/158idir/best_off_campus_housing/"},
    {"slug": "02_reddit_incoming_freshman",     "title": "r/UBreddit — incoming freshman off campus",        "url": "https://www.reddit.com/r/UBreddit/comments/1r9493z/hey_incoming_freshman_looking_for_off_campus/"},
    {"slug": "03_reddit_north_campus",          "title": "r/UBreddit — off-campus housing near North Campus", "url": "https://www.reddit.com/r/UBreddit/comments/vjvutj/offcampus_housing_near_north_campus/"},
    {"slug": "04_reddit_neighborhoods_streets", "title": "r/UBreddit — popular neighborhoods/streets",        "url": "https://www.reddit.com/r/UBreddit/comments/ibn2fg/popular_off_campus_housing_neighborhoodsstreets/"},
    {"slug": "05_reddit_no_car",                "title": "r/UBreddit — off campus housing, NO CAR",           "url": "https://www.reddit.com/r/UBreddit/comments/gu82ij/off_campus_housing_no_car/"},
    {"slug": "06_ub_off_campus_living_guide",   "title": "UB — Off-Campus Living Guide",                      "url": "https://www.buffalo.edu/community/neighbors/students/off-campus-living-guide.html"},
    {"slug": "07_ub_living_off_campus",         "title": "UB — Living Off-Campus",                            "url": "https://www.buffalo.edu/community/information-for-students/living-off-campus.html"},
    {"slug": "08_och101_portal",                "title": "UB Off-Campus Housing Portal (OCH101)",             "url": "https://buffalo.och101.com"},
    {"slug": "09_rentcollegepads",              "title": "RentCollegePads — UB off-campus search",            "url": "https://www.rentcollegepads.com/off-campus-housing/university-at-buffalo/search"},
    {"slug": "10_apartments_com",               "title": "Apartments.com — houses near North Campus",         "url": "https://www.apartments.com/off-campus-housing/ny/buffalo/state-university-of-new-york-buffalo-north-campus/houses/"},
]

# HTML tags that never hold body content.
BOILERPLATE_TAGS = ["script", "style", "nav", "header", "footer", "aside",
                    "noscript", "form", "button", "svg", "iframe", "input", "select"]

# class/id substrings that mark navigation, ads, cookie banners, share bars, etc.
BOILERPLATE_ATTR_HINTS = [
    "nav", "menu", "header", "footer", "sidebar", "breadcrumb", "cookie",
    "consent", "banner", "advert", "-ad", "ad-", "ads", "promo", "share",
    "social", "newsletter", "subscribe", "signup", "sign-up", "modal",
    "popup", "overlay", "related", "recommend", "comment-count", "vote",
    "skip-link", "masthead", "site-header", "site-footer",
]

# Short boilerplate lines to drop (matched case-insensitively, after stripping).
JUNK_LINE_PATTERNS = [
    r"^(read|show|see|view) more\b", r"^continue reading\b",
    r"^(share|tweet|reply|report|save|follow|subscribe|upvote|downvote)\b",
    r"^(sign in|log ?in|register|create account)\b",
    r"^\d+\s*(comments?|points?|upvotes?|votes?|shares?|replies)$",
    r"^skip to (main )?content$", r"^toggle navigation$",
    r"^(home|menu|search|close|back to top)$",
    r"^(privacy policy|terms( of (use|service))?|accessibility)$",
    r"^(©|copyright)\b", r"^all rights reserved",
    r"\bwe use cookies\b", r"\baccept (all )?cookies\b",
]
JUNK_LINE_RE = re.compile("|".join(JUNK_LINE_PATTERNS), re.IGNORECASE)

_tokenizer = None


def tokenizer():
    """Lazily load the embedding-compatible tokenizer (downloaded once, cached)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        # We only count tokens, never feed the model — lift the 512 cap so the
        # tokenizer doesn't spam "sequence longer than 512" warnings.
        _tokenizer.model_max_length = 10_000_000
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


# --- Stage 1: Ingestion ------------------------------------------------------
def find_local_file(slug: str) -> Path | None:
    """Return a manually-saved documents/<slug>.{json,html,htm,txt} if it exists."""
    for ext in (".json", ".html", ".htm", ".txt"):
        p = DOCUMENTS_DIR / f"{slug}{ext}"
        if p.exists():
            return p
    return None


def fetch_raw(source: dict) -> dict | None:
    """
    Build a raw record {slug, title, url, origin, raw}. Prefers a local saved
    file; falls back to fetching the URL. Returns None if neither is available.
    """
    local = find_local_file(source["slug"])
    if local is not None:
        return {**source, "origin": f"local:{local.name}", "raw": local.read_text(encoding="utf-8", errors="ignore")}

    try:
        resp = requests.get(source["url"], headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return {**source, "origin": "fetched", "raw": resp.text}
    except Exception as e:
        print(f"  ! {source['slug']}: could not fetch ({type(e).__name__}: {str(e)[:60]}). "
              f"Save the page to documents/{source['slug']}.html and re-run.")
        return None


# --- Stage 2: Cleaning -------------------------------------------------------
def _is_boilerplate_node(tag) -> bool:
    ident = " ".join(filter(None, [
        " ".join(tag.get("class", [])),
        tag.get("id", "") or "",
        tag.get("role", "") or "",
    ])).lower()
    if not any(hint in ident for hint in BOILERPLATE_ATTR_HINTS):
        return False
    # Safety valve: never delete a node that holds substantial text — a content
    # wrapper can carry a class like "share" without being a share widget.
    return len(tag.get_text(strip=True)) < 200


def looks_like_reddit_json(raw: str) -> bool:
    head = raw.lstrip()[:200]
    return head.startswith(("[", "{")) and '"kind"' in head


def _walk_comments(children: list, out: list) -> None:
    """Depth-first walk of a Reddit comment listing, appending 'author: body'."""
    for child in children:
        if child.get("kind") != "t1":
            continue
        d = child.get("data", {})
        body = (d.get("body") or "").strip()
        author = d.get("author") or "unknown"
        if body and body not in ("[deleted]", "[removed]"):
            out.append(f"Comment by {author}: {body}")
        replies = d.get("replies")
        if isinstance(replies, dict):
            _walk_comments(replies.get("data", {}).get("children", []), out)


def parse_reddit_json(raw: str) -> str:
    """Extract post + full comment tree from a Reddit thread's .json payload."""
    data = json.loads(raw)
    parts: list[str] = []
    post_children = data[0]["data"]["children"]
    if post_children:
        p = post_children[0]["data"]
        title = (p.get("title") or "").strip()
        author = p.get("author") or "unknown"
        selftext = (p.get("selftext") or "").strip()
        parts.append(f"Post title: {title}")
        parts.append(f"Original post by {author}: {selftext}".strip(": ")
                     if selftext else f"Original post by {author}.")
    if len(data) > 1:
        _walk_comments(data[1]["data"]["children"], parts)
    return "\n\n".join(parts)


def clean_document(raw: str) -> str:
    """Turn raw HTML / Reddit JSON / text into clean, domain-relevant plain text."""
    if looks_like_reddit_json(raw):
        text = parse_reddit_json(raw)            # already clean structured text
    elif "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(BOILERPLATE_TAGS):
            tag.decompose()
        for tag in soup.find_all(_is_boilerplate_node):
            tag.decompose()
        root = soup.body or soup
        text = root.get_text(separator="\n")
    else:
        text = raw

    text = html.unescape(text)                  # &amp; -> &, &nbsp; -> \xa0, ...
    text = text.replace("\xa0", " ")            # non-breaking spaces -> spaces

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        # Drop short boilerplate lines; keep anything substantial (likely content).
        if len(line) < 60 and JUNK_LINE_RE.search(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n\n", text)      # collapse blank-line runs
    return text.strip()


# --- Stage 2: Chunking (token-based, paragraph-aware) ------------------------
def split_sentences(paragraph: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", paragraph) if p.strip()]


def hard_split_by_tokens(unit: str, max_tokens: int) -> list[str]:
    """Split an oversized unit into <=max_tokens pieces using the tokenizer."""
    tok = tokenizer()
    ids = tok.encode(unit, add_special_tokens=False)
    pieces = []
    for i in range(0, len(ids), max_tokens):
        pieces.append(tok.decode(ids[i:i + max_tokens]).strip())
    return [p for p in pieces if p]


def split_into_units(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Break text into (unit, token_count) no larger than max_tokens each."""
    units: list[tuple[str, int]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append((para, count_tokens(para)))
            continue
        for sentence in split_sentences(para):
            if count_tokens(sentence) <= max_tokens:
                units.append((sentence, count_tokens(sentence)))
            else:
                for piece in hard_split_by_tokens(sentence, max_tokens):
                    units.append((piece, count_tokens(piece)))
    return units


def chunk_text(doc_text: str, max_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """
    Pack paragraph/sentence units into <=max_tokens chunks, re-including trailing
    units so neighbors overlap by ~overlap_tokens. Overlap keeps a fact that sits
    on a boundary retrievable from a single chunk.
    """
    units = split_into_units(doc_text, max_tokens)
    chunks: list[str] = []
    start = 0

    while start < len(units):
        cur: list[str] = []
        cur_tokens = 0
        end = start
        while end < len(units):
            _, tcount = units[end]
            if cur and cur_tokens + tcount > max_tokens:
                break
            cur.append(units[end][0])
            cur_tokens += tcount
            end += 1

        chunks.append("\n\n".join(cur))

        if end >= len(units):
            break

        # Walk back from `end` to gather ~overlap_tokens of trailing units.
        overlap_acc = 0
        next_start = end
        j = end - 1
        while j > start and overlap_acc + units[j][1] <= overlap_tokens:
            overlap_acc += units[j][1]
            next_start = j
            j -= 1

        start = next_start  # always > old start, so we make progress

    return chunks


# --- Driver ------------------------------------------------------------------
def ingest_raw() -> list[dict]:
    """Load all sources and persist raw text to raw/raw_documents.jsonl."""
    RAW_DIR.mkdir(exist_ok=True)
    records = []
    for source in SOURCES:
        rec = fetch_raw(source)
        if rec is None:
            continue
        records.append(rec)
        print(f"  - {rec['slug']:32s} [{rec['origin']:18s}] {len(rec['raw']):>8,} raw chars")

    with RAW_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} raw documents to {RAW_FILE}")
    return records


def main() -> None:
    print("=== Stage 1: Ingestion ===")
    raw_records = ingest_raw()
    if not raw_records:
        print("\nNo documents ingested. Save pages into documents/ and re-run.")
        return

    print("\n=== Stage 2: Cleaning + Chunking "
          f"(size={CHUNK_TOKENS} tok, overlap={OVERLAP_TOKENS} tok) ===")
    all_chunks: list[dict] = []
    cleaned_by_slug: dict[str, str] = {}
    for rec in raw_records:
        cleaned = clean_document(rec["raw"])
        cleaned_by_slug[rec["slug"]] = cleaned
        if not cleaned:
            print(f"  ! {rec['slug']}: no text after cleaning, skipping")
            continue
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{rec['slug']}::{i}",
                "source": rec["title"],
                "slug": rec["slug"],
                "url": rec["url"],
                "chunk_index": i,
                "token_count": count_tokens(chunk),
                "char_count": len(chunk),
                "text": chunk,
            })
        print(f"  - {rec['slug']:32s} {len(cleaned):>8,} clean chars -> {len(chunks)} chunks")

    CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Inspect one cleaned document --------------------------------------
    first_slug = next(iter(cleaned_by_slug))
    sample_doc = cleaned_by_slug[first_slug]
    print("\n=== Inspect one cleaned document:", first_slug, "===")
    print(sample_doc[:2000])
    if len(sample_doc) > 2000:
        print(f"... [truncated, {len(sample_doc):,} chars total]")

    # --- Inspect 5 representative chunks (spread across documents) ----------
    print("\n=== 5 representative chunks ===")
    if all_chunks:
        step = max(1, len(all_chunks) // 5)
        samples = all_chunks[::step][:5]
        for c in samples:
            print(f"\n--- {c['id']}  | source: {c['source']}  | {c['token_count']} tokens ---")
            print(c["text"][:700] + ("..." if len(c["text"]) > 700 else ""))

    # --- Summary + sanity check --------------------------------------------
    total = len(all_chunks)
    print(f"\n=== Summary ===\nTotal chunks: {total} across {len(raw_records)} documents")
    if all_chunks:
        toks = [c["token_count"] for c in all_chunks]
        print(f"tokens/chunk: min={min(toks)} avg={sum(toks)//len(toks)} max={max(toks)}")
    if total < 50:
        print("WARNING: < 50 chunks — chunks may be too large, or sources are missing "
              "(blocked Reddit/apartments.com pages need to be saved into documents/).")
    elif total > 2000:
        print("WARNING: > 2,000 chunks — chunks may be too small.")
    else:
        print("Chunk count is within the healthy 50-2,000 range.")
    print(f"\nWrote {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
