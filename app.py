from flask import Flask, render_template, request, send_file
from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber
import os
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# ---------------- MODEL ----------------
MODEL_PATH = "pdf_summarizer_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)


# ---------------- PDF ----------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- CHUNKING ----------------
def adaptive_chunking(text, tokenizer, max_tokens=900):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    
    for sentence in sentences:
        if len(tokenizer(current_chunk + sentence)["input_ids"]) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------- RANKING ----------------
def rank_chunks(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    scores = np.sum(X.toarray(), axis=1)
    ranked = [c for _, c in sorted(zip(scores, chunks), reverse=True)]
    return ranked


# ---------------- SELECTION ----------------
def select_chunks(ranked_chunks, mode):
    total = len(ranked_chunks)

    if mode == "short":
        return ranked_chunks[:max(2, int(0.2 * total))]
    elif mode == "detailed":
        return ranked_chunks[:max(8, int(0.8 * total))]
    else:
        return ranked_chunks[:max(4, int(0.5 * total))]


# ---------------- CHUNK SUMMARY ----------------
def summarize_chunks(chunks, mode):
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)

        ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4
        )

        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    return summaries


# ---------------- CLEANING ----------------
def remove_redundancy(text):
    sentences = text.split(".")
    seen = set()
    unique = []

    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    return ". ".join(unique)


# ---------------- FINAL REFINEMENT ----------------
def refine_summary(text, mode):
    if mode == "short":
        max_len, min_len = 80, 30
    elif mode == "detailed":
        max_len, min_len = 300, 120
    else:
        max_len, min_len = 150, 60

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=6,
        length_penalty=2.2,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------- TITLE ----------------
def generate_title(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    ids = model.generate(inputs["input_ids"], max_length=20)

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------- KEY POINTS ----------------
def extract_key_points(text):
    sentences = text.split(".")
    points = []

    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            points.append(s)
        if len(points) == 5:
            break

    return points


# ---------------- MAIN PIPELINE ----------------
def advanced_summarization(text, mode):
    chunks = adaptive_chunking(text, tokenizer)

    ranked = rank_chunks(chunks)
    selected = select_chunks(ranked, mode)

    chunk_summaries = summarize_chunks(selected, mode)

    combined = " ".join(chunk_summaries)
    combined = remove_redundancy(combined)

    final_summary = refine_summary(combined, mode)
    final_summary = remove_redundancy(final_summary)

    title = generate_title(final_summary)
    points = extract_key_points(final_summary)

    return {
        "title": title,
        "summary": final_summary,
        "points": points
    }


# ---------------- PDF SAVE ----------------
def save_summary_pdf(text, filename="summary.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750

    for line in text.split("\n"):
        c.drawString(40, y, line[:90])
        y -= 15
        if y < 40:
            c.showPage()
            y = 750

    c.save()


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["pdf"]
        mode = request.form["mode"]

        if file:
            path = "temp.pdf"
            file.save(path)

            text = extract_text_from_pdf(path)
            text = clean_text(text)

            result = advanced_summarization(text, mode)

            save_summary_pdf(result["summary"])

            os.remove(path)

    return render_template("index.html", result=result)


@app.route("/download")
def download():
    return send_file("summary.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)