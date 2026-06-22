"""
query.py — Stage 6 of the RAG pipeline: Grounded Generation.

ask(question) retrieves the top-k chunks (retrieve.py), builds a strictly-grounded
prompt, and asks Groq's llama-3.3-70b-versatile to answer using ONLY that context.
Source attribution is handled two ways for safety:
  1. the model is instructed to cite the [source: ...] labels inline, AND
  2. the unique source documents of the retrieved chunks are returned
     programmatically in result["sources"] — so attribution is guaranteed
     regardless of what the model writes.

As a library:
    from query import ask
    result = ask("Is $750/month enough near UB?")
    print(result["answer"]); print(result["sources"])

As a script (runs grounded-generation tests, incl. an out-of-domain query):
    .venv/bin/python query.py
"""

import os

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve, DEFAULT_K

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
NO_INFO = "I don't have enough information on that."

# Grounding is ENFORCED here, not suggested: the model is told to use only the
# context, to refuse with an exact sentence when the context is insufficient,
# and never to fall back on outside knowledge.
SYSTEM_PROMPT = f"""You are a retrieval-grounded assistant that answers questions about \
off-campus housing for University at Buffalo (UB) students.

Rules — follow them exactly:
1. Answer ONLY using the information in the CONTEXT documents provided in the user message.
2. Do NOT use any prior or outside knowledge. If the answer is not contained in the CONTEXT, \
you MUST reply with exactly this sentence and nothing else: "{NO_INFO}"
3. Do not guess, infer beyond the text, or add general advice that is not supported by the CONTEXT.
4. Cite your sources inline using the bracketed [source: ...] labels exactly as they appear \
above each document, e.g. (source: r/UBreddit — best off campus housing).
5. Keep the answer concise and specific to what the documents actually say."""

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only the CONTEXT above, citing the [source: ...] labels. If the CONTEXT does \
not contain enough information to answer, reply exactly: "{no_info}"."""


def _format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as numbered, source-labeled blocks for the prompt."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"Document {i} (source: {h['source']})\n{h['text']}")
    return "\n\n".join(blocks)


def _unique_sources(hits: list[dict]) -> list[dict]:
    """Distinct source documents (name + url), preserving retrieval order."""
    seen, sources = set(), []
    for h in hits:
        if h["source"] not in seen:
            seen.add(h["source"])
            sources.append({"source": h["source"], "url": h["url"]})
    return sources


def ask(question: str, k: int = DEFAULT_K) -> dict:
    """
    Retrieve top-k chunks and generate a grounded answer.
    Returns {answer, sources, chunks} where sources is the programmatic
    attribution (list of {source, url}) and chunks are the retrieved hits.
    """
    hits = retrieve(question, k=k)
    if not hits:
        return {"answer": NO_INFO, "sources": [], "chunks": []}

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # deterministic + faithful to context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=_format_context(hits),
                question=question,
                no_info=NO_INFO,
            )},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # If the model refused (no info in context), don't attach misleading sources.
    sources = [] if answer.strip().rstrip(".") == NO_INFO.rstrip(".") else _unique_sources(hits)
    return {"answer": answer, "sources": sources, "chunks": hits}


# --- End-to-end grounded-generation tests ------------------------------------
TEST_QUERIES = [
    "What do students think about popular housing complexes near North Campus?",
    "Is a budget of $750 per month enough for off-campus housing near UB?",
    "How do students commute to North Campus without a car?",
    # Out-of-domain: documents don't cover this -> system must refuse.
    "What is the best pizza topping in Buffalo?",
]


def _run_tests() -> None:
    for q in TEST_QUERIES:
        result = ask(q)
        print("\n" + "=" * 100)
        print("Q:", q)
        print("-" * 100)
        print("ANSWER:\n", result["answer"])
        if result["sources"]:
            print("\nSOURCES (programmatic):")
            for s in result["sources"]:
                print(f"  • {s['source']} — {s['url']}")
        else:
            print("\nSOURCES: (none — out-of-context / refusal)")
        print("\nRETRIEVAL distances:",
              ", ".join(f"{h['distance']:.3f}" for h in result["chunks"]))


if __name__ == "__main__":
    _run_tests()
