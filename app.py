import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber
import torch
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string



def setup_nltk():
    import os
    nltk_data_path = "/tmp/nltk_data"
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    resources = ["punkt", "punkt_tab", "stopwords"]

    for r in resources:
        try:
            if r == "stopwords":
                nltk.data.find("corpora/stopwords")
            else:
                nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r, download_dir=nltk_data_path)

setup_nltk()


MODEL_PATH = "vatsal0025/DL_pdf_summarizer_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)


# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def clean_text(text):
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'http\S+', '', text)

    # 🔥 FIX BROKEN WORDS
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)

    # 🔥 REMOVE TABLE LINES
    text = re.sub(r'TABLE\s*\w+.*', '', text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def adaptive_chunking(text):
    paragraphs = text.split("\n")
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    chunks = []
    current_chunk = []
    current_length = 0
    max_tokens = 800

    for para in paragraphs:
        sentences = sent_tokenize(para)
        for sentence in sentences:
            sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def rank_chunks(chunks):
    if not chunks:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    scores = np.sum(X.toarray(), axis=1)

    ranked = [(i, c) for _, i, c in sorted(zip(scores, range(len(chunks)), chunks), reverse=True)]
    return ranked


def select_chunks(ranked_chunks, mode):
    total = len(ranked_chunks)

    if mode == "short":
        count = max(1, int(0.2 * total))
    elif mode == "detailed":
        count = max(5, int(0.6 * total))
    else:
        count = max(3, int(0.4 * total))

    selected = ranked_chunks[:count]
    selected.sort(key=lambda x: x[0])
    return [c for i, c in selected]


def summarize_chunks(chunks, mode):
    summaries = []

    params = {
        "short": (80, 20),
        "medium": (150, 40),
        "detailed": (300, 80)
    }

    max_len, min_len = params[mode]

    for chunk in chunks:
        inputs = tokenizer("summarize: " + chunk,
                           return_tensors="pt",
                           truncation=True,
                           max_length=1024).to(device)

        ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    return summaries


def refine_summary(text, mode):
    params = {
        "short": (120, 30),
        "medium": (250, 60),
        "detailed": (400, 80)
    }
    max_len, min_len = params[mode]
    
    # Dynamically adjust min_length to prevent hallucination on short texts
    input_length = len(text.split())
    if input_length < min_len:
        min_len = max(5, input_length // 2)

    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


def generate_title(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    ids = model.generate(inputs["input_ids"], max_length=15, min_length=2, num_beams=4, early_stopping=True)
    title = tokenizer.decode(ids[0], skip_special_tokens=True)
    
    # Strip dangling trailing words to prevent hard cut-offs
    title = re.sub(r'\s+(and|or|the|in|with|a|an|of|,|:)$', '', title.strip(), flags=re.IGNORECASE)
    return title.rstrip(".,;: ")



def clean_sentence(s):
    s = s.strip()

    # ❌ remove table-like or broken text
    if len(s) > 200:
        return False

    if any(x in s.lower() for x in [
        "table", "fig", "figure", "arxiv", "doi", "www", "http"
    ]):
        return False

    # ❌ too many uppercase → likely table/caption
    if sum(1 for c in s if c.isupper()) > len(s) * 0.4:
        return False

    # ❌ too many numbers → likely data row
    if sum(c.isdigit() for c in s) > 5:
        return False

    # ❌ no spaces (broken extraction)
    if len(s.split()) < 5:
        return False

    return True
    
def extract_key_points_nltk(text, num_points=5):

    sentences = list(set(sent_tokenize(text)))

    # remove weak sentences
    sentences = [s for s in sentences if clean_sentence(s)]

    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    if not freq:
        return sentences[:num_points]

    max_freq = max(freq.values())
    for word in freq:
        freq[word] /= max_freq

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freq[word]

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # 🔥 avoid duplicates / similar sentences
    selected = []
    for sent in ranked_sentences:
        if len(selected) >= num_points:
            break

        if all(sent[:40] not in s for s in selected):
            selected.append(sent)

    return selected


def save_pdf(title, points_text, summary_text):
    filename = "summary.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750

    def draw_text(text, is_bold=False):
        nonlocal y
        if is_bold:
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 11)
            
        for line in text.split("\n"):
            # Simple text wrapping
            words = line.split(" ")
            current_line = ""
            for word in words:
                if len(current_line) + len(word) < 90:
                    current_line += word + " "
                else:
                    c.drawString(40, y, current_line.strip())
                    y -= 15
                    if y < 40:
                        c.showPage()
                        y = 750
                    current_line = word + " "
            if current_line:
                c.drawString(40, y, current_line.strip())
                y -= 15
                if y < 40:
                    c.showPage()
                    y = 750

    draw_text("TITLE", is_bold=True)
    draw_text(title)
    y -= 20
    
    draw_text("KEY POINTS", is_bold=True)
    draw_text(points_text.strip())
    y -= 20
    
    draw_text("SUMMARY", is_bold=True)
    draw_text(summary_text)

    c.save()
    return filename


# ---------------- MAIN PIPELINE ----------------
def process(file, mode):
    text = extract_text_from_pdf(file)
    text = clean_text(text)

    chunks = adaptive_chunking(text)
    ranked = rank_chunks(chunks)
    selected = select_chunks(ranked, mode)

    chunk_summaries = summarize_chunks(selected, mode)
    combined = " ".join(chunk_summaries)

    final_summary = refine_summary(combined, mode)
    title = generate_title(final_summary)
    
    points = extract_key_points_nltk(" ".join(selected), num_points=4)

    # 🔥 FORMAT POINTS CLEANLY
    cleaned_points = []
    for p in points:
        if len(p) > 130:
            # truncate at nearest space to avoid broken words
            trunc = p[:130]
            last_space = trunc.rfind(" ")
            if last_space > -1:
                trunc = trunc[:last_space]
            p = trunc + "..."
        cleaned_points.append(p)
        
    points_text = "\n• " + "\n• ".join(cleaned_points)
    pdf_path = save_pdf(title, points_text, final_summary)
    

    return title,points_text, final_summary, pdf_path


# ---------------- PREMIUM UI ----------------
with gr.Blocks() as app:
    gr.Markdown("""
    # 📄 Smart PDF Summarizer  

    Upload any PDF and get:
    - 📌 Title  
    - 🔑 Key Insights  
    - 📝 Clean Summary  
    """)

    with gr.Row():
        file_input = gr.File(label="📂 Upload PDF", file_types=[".pdf"])
        mode = gr.Radio(
            ["short", "medium", "detailed"],
            value="medium",
            label="📊 Summary Depth"
        )

    submit = gr.Button("✨ Generate Summary")

    with gr.Row():
        title_box = gr.Textbox(label="📌 Title", placeholder="Title will appear here...")
    
    with gr.Row():
        points_box = gr.Textbox(label="🔑 Key Points", placeholder="Key insights will appear here...")
    
    with gr.Row():
        summary_box = gr.Textbox(label="📝 Summary", placeholder="Summary will appear here...")

    with gr.Row():
        download_file = gr.File(label="⬇️ Download Summary PDF", interactive=False)

    status = gr.Markdown("")

    # -------- LOADING + OUTPUT --------
    def run_pipeline(file, mode):
        try:
            if file is None:
                return "", "", "", None, "❌ Please upload a PDF"

            title, points, summary, pdf_path = process(file, mode)

            return title, points, summary, pdf_path, "✅ Done!"

        except Exception as e:
            print("ERROR:", str(e))
            return "Error", "Error", str(e), None, "❌ Failed"
        
    submit.click(
        run_pipeline,
        inputs=[file_input, mode],
        outputs=[title_box, points_box, summary_box, download_file, status]
    )



app.launch(theme=gr.themes.Soft(primary_hue="indigo"))