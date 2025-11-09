import os
import streamlit as st
from transformers.pipelines import pipeline
from bs4 import BeautifulSoup
import requests

st.set_page_config(page_title="Summarize Anything", page_icon="🧩", layout="centered")
st.title("🧩 SummaYara")
st.markdown("### Your friendly AI that turns chaos into clarity")
st.caption("Built with open-source models. Made by Yara.")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch_text_from_url(url: str) -> str:
    """Fetch readable text from a URL (trafilatura -> Wikipedia render -> generic BS4)."""
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
    except Exception:
        return ""

    
    try:
        import trafilatura
        extracted = trafilatura.extract(html, favor_recall=True, include_comments=False)
        if extracted and len(extracted) > 200:
            return extracted
    except Exception:
        pass

    # Wikipedia
    if "wikipedia.org" in url:
        sep = "&" if "?" in url else "?"
        wiki_url = f"{url}{sep}action=render"
        try:
            html2 = requests.get(wiki_url, headers=HEADERS, timeout=15).text
            soup2 = BeautifulSoup(html2, "html.parser")
            content2 = soup2.find("div", {"class": "mw-parser-output"})
            if content2:
                blocks2 = content2.find_all(["p", "li"])
                text2 = "\n".join(b.get_text(" ", strip=True) for b in blocks2)
                if len(text2) > 200:
                    return text2
        except Exception:
            pass

    # Generic BeautifulSoup fallback
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("article") or soup.find("main") or soup.find("div", {"id": "mw-content-text"}) or soup
    blocks = main.find_all(["p", "li"])
    text = "\n".join(b.get_text(" ", strip=True) for b in blocks)
    return text if len(text) > 200 else ""

# MODEL
@st.cache_resource(show_spinner=True)
def load_summarizer():
    # Always use CPU to avoid MPS issues on Mac
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1,  # CPU
    )

summarizer = load_summarizer()

with st.expander("Tips"):
    st.write("- Paste 1–10 paragraphs of clean text.\n- Keep 'Chunk long text' on for very long inputs.")

# UI
url_mode = st.checkbox("Fetch text from a URL", value=False)

text = ""
if url_mode:
    url = st.text_input("Paste a URL:")
    if st.button("Fetch"):
        with st.spinner("Fetching and cleaning page…"):
            extracted = fetch_text_from_url(url)
            if extracted:
                st.success("Fetched! Loaded into the text box.")
                text = extracted
                with st.expander("Preview fetched text"):
                    st.text_area("Extracted text preview:", value=text[:2000], height=200)
            else:
                st.warning("Couldn’t extract enough text from that page.")

text = st.text_area("Paste text to summarize:", value=text, height=220, placeholder="Paste an article or notes here…")

col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Max summary length (tokens)", 60, 200, 120, step=10)
with col2:
    min_len = st.slider("Min summary length (tokens)", 20, 120, 40, step=10)

chunking = st.checkbox("Chunk long text automatically", value=True)

def chunk_text(t, max_chars=2500):
    t = t.strip()
    if len(t) <= max_chars:
        return [t]
    chunks, start = [], 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        cut = t.rfind(".", start, end)
        if cut == -1 or cut <= start + 500:
            cut = end
        else:
            cut += 1
        chunks.append(t[start:cut].strip())
        start = cut
    return chunks

if st.button("Summarize", type="primary", disabled=(len(text.strip()) == 0)):
    with st.spinner("Summarizing…"):
        try:
            if chunking:
                pieces = chunk_text(text)
                partials = []
                for p in pieces:
                    s = summarizer(
                        p,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False,
                        num_beams=2,
                        truncation=True,
                    )[0]["summary_text"]
                    partials.append(s)
                final = summarizer(
                    " ".join(partials),
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    num_beams=2,
                    truncation=True,
                )[0]["summary_text"]
                st.success(final)
                with st.expander("Show intermediate chunk summaries"):
                    for i, s in enumerate(partials, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(s)
            else:
                summary = summarizer(
                    text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    num_beams=2,
                    truncation=True,
                )[0]["summary_text"]
                st.success(summary)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Tech: Streamlit + Hugging Face Transformers • Model: sshleifer/distilbart-cnn-12-6 (CPU)")
