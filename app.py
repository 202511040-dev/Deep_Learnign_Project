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
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

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
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)  # remove citations [1,2]
    text = re.sub(r'\(.*?\)', '', text)           # remove brackets (..)
    text = re.sub(r'http\S+', '', text)           # remove URLs
    
    # Remove common academic paper boilerplate that pollutes key points
    text = re.sub(r'Authorized licensed use limited to:.*?Restrictions apply\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'IEEE Xplore\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©.*?IEEE', '', text)
    text = re.sub(r'in Proc\..*?\d{4}', '', text)
    text = re.sub(r'\s+', ' ', text)              # normalize spaces
    return text.strip()


# ---------------- CHUNKING ----------------
def split_by_paragraph(text):
    paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]

def get_dynamic_max_length(text):
    length = len(text.split())
    if length < 2000:
        return 800
    elif length < 5000:
        return 900
    else:
        return 1000

def adaptive_chunking(text, tokenizer):
    paragraphs = split_by_paragraph(text)
    max_tokens = get_dynamic_max_length(text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        sentences = sent_tokenize(para)
        for sentence in sentences:
            sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # smarter overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------- RANKING ----------------
def rank_chunks(chunks):
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    scores = np.sum(X.toarray(), axis=1)
    
    # Normalize by chunk length to prevent bias towards longer chunks
    normalized_scores = []
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        normalized_score = scores[i] / (word_count + 1e-5)
        normalized_scores.append(normalized_score)
        
    ranked = [(i, c) for _, i, c in sorted(zip(normalized_scores, range(len(chunks)), chunks), reverse=True)]
    return ranked


# ---------------- SELECTION ----------------
def select_chunks(ranked_chunks, mode):
    total = len(ranked_chunks)

    if mode == "short":
        count = min(3, max(1, int(0.15 * total)))
    elif mode == "detailed":
        count = min(15, max(5, int(0.6 * total)))
    else:
        count = min(6, max(3, int(0.25 * total)))
        
    top_chunks = ranked_chunks[:count]
    # Sort by original index to maintain chronological order and coherent meaning
    top_chunks.sort(key=lambda x: x[0])
    
    return [c for i, c in top_chunks]


# ---------------- CHUNK SUMMARY ----------------
def get_summary_params(mode):
    if mode == "short":
        return {"max_length": 80, "min_length": 15, "num_beams": 2}
    elif mode == "detailed":
        return {"max_length": 250, "min_length": 30, "num_beams": 3}
    else:  # medium
        return {"max_length": 150, "min_length": 20, "num_beams": 2}

def summarize_chunks(chunks, mode):
    summaries = []
    params = get_summary_params(mode)

    for chunk in chunks:
        chunk_text = "summarize: " + chunk
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=params["max_length"],
            min_length=params["min_length"],
            num_beams=params["num_beams"],
            length_penalty=1.5,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
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
        max_len, min_len, beams = 100, 20, 2
    elif mode == "detailed":
        max_len, min_len, beams = 350, 40, 3
    else:
        max_len, min_len, beams = 200, 30, 2

    text_input = "summarize: " + text
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024).to(device)

    ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=max_len,
        min_length=min_len,
        num_beams=beams,
        length_penalty=1.5,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------- TITLE ----------------
def generate_title(text):
    text_input = "summarize: " + text
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512).to(device)

    ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs.get("attention_mask"),
        max_length=20,
        min_length=5,
        num_beams=2,
        early_stopping=True
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------- KEY POINTS ----------------
def extract_key_points(text, num_points=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_points:
        return sentences

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(sentences)
        scores = np.sum(X.toarray(), axis=1)
        
        # Normalize by length to prevent bias towards massive sentences
        normalized_scores = [s / (len(sent.split()) + 1e-5) for s, sent in zip(scores, sentences)]
        
        ranked = [s for _, s in sorted(zip(normalized_scores, sentences), reverse=True)]
        
        # Filter for sentences with reasonable length for a key point
        valid_points = [s for s in ranked if 40 < len(s) < 200]
        if not valid_points:
            valid_points = ranked
            
        return valid_points[:num_points]
    except ValueError:
        return sentences[:num_points]


# ---------------- MAIN PIPELINE ----------------
def advanced_summarization(text, mode):
    chunks = adaptive_chunking(text, tokenizer)

    ranked = rank_chunks(chunks)
    selected = select_chunks(ranked, mode)

    chunk_summaries = summarize_chunks(selected, mode)

    # For detailed mode, separate with double newlines for structure
    if mode == "detailed":
        combined = "\n\n".join(chunk_summaries)
        final_summary = combined
    else:
        combined = " ".join(chunk_summaries)
        combined = remove_redundancy(combined)
        final_summary = refine_summary(combined, mode)
        final_summary = remove_redundancy(final_summary)

    title = generate_title(final_summary)
    
    # Extract key points from the original text (selected chunks) to avoid duplication with summary
    original_text = " ".join(selected)
    points = extract_key_points(original_text)

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