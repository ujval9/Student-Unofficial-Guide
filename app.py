"""
app.py — Stage 7 of the RAG pipeline: Query Interface.

A minimal Gradio web UI over the grounded-generation function in query.py.
Type a housing question, get a grounded answer plus the source documents it
was retrieved from.

Run:
    .venv/bin/python app.py
Then open http://localhost:7860
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "What do students think about popular housing complexes near North Campus?",
    "Is a budget of $750 per month enough for off-campus housing near UB?",
    "How do students commute to North Campus without a car?",
    "What should I check before signing an off-campus lease?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s['source']} — {s['url']}" for s in result["sources"])
    return result["answer"], sources or "(no sources — outside the document set)"


with gr.Blocks(title="UB Off-Campus Housing — Unofficial Guide") as demo:
    gr.Markdown(
        "# UB Off-Campus Housing — Unofficial Guide\n"
        "Ask about neighborhoods, complexes, rent, commuting, and leases near "
        "University at Buffalo. Answers are grounded **only** in retrieved student "
        "discussions and UB guides — with sources shown."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. Is $750/month enough near UB?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
