"""
Book Recommendation System — Gradio Frontend
Run:  python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import recommender as rc
import time


# ─── Boot ────────────────────────────────────────────────────────────────────
_status = rc.load_and_train("books.csv")
_READY  = _status["ok"]
_TITLES = rc.get_all_titles() if _READY else []

STAR = "⭐"
LANG_FLAG = {
    "eng": "🇬🇧", "en-US": "🇺🇸", "en-GB": "🇬🇧", "fre": "🇫🇷",
    "ger": "🇩🇪", "spa": "🇪🇸", "por": "🇵🇹", "ita": "🇮🇹",
    "jpn": "🇯🇵", "zho": "🇨🇳", "ara": "🇸🇦", "rus": "🇷🇺",
}

def stars(rating: float) -> str:
    full = int(round(rating))
    return STAR * full + "☆" * (5 - full) + f"  {rating:.1f}"


def format_results(recs: list[dict]) -> str:
    if not recs:
        return ""
    html = []
    for i, r in enumerate(recs, 1):
        flag = LANG_FLAG.get(r["language_code"], "📖")
        count = f"{r['ratings_count']:,}"
        html.append(
            f'<div class="result-card">\n'
            f'  <h3 style="margin-top:0; color:#6366f1">{i}. {r["title"]}</h3>\n'
            f'  <p style="margin-bottom:8px">✍️ <b>{r["authors"]}</b> • {stars(r["average_rating"])}</p>\n'
            f'  <p style="font-size:0.9rem; color:#94a3b8">{count} ratings • {flag} <code>{r["language_code"]}</code></p>\n'
            f'</div>'
        )
    return "\n".join(html)


def search_books(query: str, n_recs: int):
    """Main callback."""
    query = (query or "").strip()
    if not query:
        return "⚠️ Please enter a book title.", gr.update()

    if not _READY:
        return f"❌ {_status['message']}", gr.update()

    # Log query with timestamp
    print(f"\n[{time.strftime('%H:%M:%S')}] UI: Requesting recommendations for '{query}'...")
    gr.Info(f"🔍 Finding books similar to '{query}'...")

    recs = rc.recommend(query, n=int(n_recs))
    if not recs:
        return (
            f"❌ **'{query}'** not found in the dataset.\n\n"
            "Try selecting a title from the dropdown or check your spelling.",
            gr.update(),
        )

    summary = f"### 📚 Books similar to *{query}*\n\n"
    return summary, format_results(recs)


# ─── UI Styling ──────────────────────────────────────────────────────────────
CSS = """
/* Refined Slate Theme */
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important;
    min-height: 100vh !important;
    color: #f1f5f9 !important;
}

.glass-card {
    background: rgba(15, 23, 42, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
    padding: 30px !important;
    position: relative !important;
    z-index: 10 !important;
}

.search-btn {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: white !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.search-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4) !important;
}

/* Fix Label Visibility & Container Contrast */
.dropdown-container, .gr-box, div.container.svelte-1hguek3, div.container.svelte-8epfm4 {
    background: #0f172a !important; /* Deep Navy */
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
}

label span, 
.block-label span,
.svelte-1hguek3 label span,
.svelte-8epfm4 label span {
    color: #a78bfa !important; /* Bright Lavender/Violet */
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.85rem !important;
    margin-bottom: 6px !important;
    display: block !important;
}

ul.options {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    max-height: 280px !important;
    overflow-y: auto !important;
    z-index: 1001 !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6) !important;
}

li.item {
    color: #f1f5f9 !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}

li.item:hover, li.item.selected {
    background: #334155 !important;
    color: #ffffff !important;
}

/* Enhanced tactile scrollbar */
::-webkit-scrollbar {
    width: 12px !important;
}
::-webkit-scrollbar-track {
    background: #0f172a !important;
}
::-webkit-scrollbar-thumb {
    background: #475569 !important;
    border-radius: 6px !important;
    border: 3px solid #0f172a !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #6366f1 !important;
}

/* Result Cards */
.result-card {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-left: 5px solid #6366f1 !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.result-card:hover {
    transform: translateX(8px) !important;
    background: #243049 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #8b5cf6, #6366f1)",
    button_primary_background_fill_hover="linear-gradient(90deg, #7c3aed, #4f46e5)",
    block_radius="16px",
)

with gr.Blocks(title="📚 Book Recommender") as demo:

    with gr.Column(elem_classes=["glass-card"], scale=1):
        gr.HTML(
            """
            <div style="text-align:center; padding: 20px 0">
              <h1 style="font-size:3rem; font-weight:800; margin-bottom:8px; background: linear-gradient(90deg, #8b5cf6, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📚 Book Recommender
              </h1>
              <p style="color:#64748b; font-size:1.1rem; max-width:600px; margin: 0 auto">
                Discover your next favorite read using cutting-edge KNN analysis across 11,000+ literary masterpieces.
              </p>
            </div>
            """
        )

        if not _READY:
            gr.Markdown(
                f"""
                > ⚠️ **Dataset not found.**  
                > Place `books.csv` in the same folder as `app.py` and restart.  
                > Download from [Kaggle – Goodreads Books](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks).
                """,
                elem_id="error-msg"
            )
        else:
            gr.Markdown(
                f"<p style='text-align:center; color:#8b5cf6; font-weight:500'>✨ Model Ready — browsing {len(_TITLES):,} books</p>"
            )

        with gr.Row():
            with gr.Column(scale=3):
                book_input = gr.Textbox(
                    placeholder="e.g. Harry Potter, The Hobbit, 1984...",
                    label="🔍 Search for a Book",
                    info="Type the title of a book you enjoyed",
                    lines=1,
                )
            with gr.Column(scale=1):
                n_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Count",
                )

        search_btn = gr.Button("✨ Find Recommendations", variant="primary", size="lg", elem_classes=["search-btn"])

    with gr.Column(visible=False) as results_area:
        header_md = gr.Markdown()
        results_md = gr.Markdown()

    if _READY and _TITLES:
        sample = [
            "Harry Potter and the Half-Blood Prince (Harry Potter  #6)",
            "The Da Vinci Code (Robert Langdon  #2)",
            "The Hobbit",
            "To Kill a Mockingbird",
            "1984",
        ]
        valid_samples = [t for t in sample if t in _TITLES][:4]
        if valid_samples:
            gr.Examples(
                examples=[[t, 5] for t in valid_samples],
                inputs=[book_input, n_slider],
                outputs=[header_md, results_md],
                fn=search_books,
                label="💡 Featured Examples",
                cache_examples=False,
            )

    def on_search(query, n):
        h, r = search_books(query, n)
        return gr.update(visible=True), h, r

    search_btn.click(
        fn=on_search,
        inputs=[book_input, n_slider],
        outputs=[results_area, header_md, results_md],
    )

    gr.HTML(
        """
        <div style="text-align:center; margin-top: 40px; padding: 20px; color: #94a3b8; font-size: 0.9rem; border-top: 1px solid rgba(0,0,0,0.05)">
          Built with scikit-learn KNN · Gradio · 
          Dataset: <a href="https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks" target="_blank" style="color:#8b5cf6">Goodreads (Kaggle)</a>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=THEME, css=CSS)
