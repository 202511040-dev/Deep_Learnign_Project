from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber
import os
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file

app = Flask(__name__)

# Load model
MODEL_PATH = "pdf_summarizer_model"

tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)


# ---------- YOUR EXISTING FUNCTIONS ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    import re
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def summarize(text, mode="medium"):
    params = {
        "short": (80, 30),
        "medium": (150, 60),
        "detailed": (250, 100)
    }

    max_len, min_len = params.get(mode, (150, 60))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)




def save_summary_pdf(text, filename="summary.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    lines = text.split("\n")
    y = height - 40

    for line in lines:
        c.drawString(40, y, line[:90])
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40

    c.save()

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""

    if request.method == "POST":
        file = request.files["pdf"]
        mode = request.form["mode"]

        if file:
            file_path = "temp.pdf"
            file.save(file_path)

            text = extract_text_from_pdf(file_path)
            text = clean_text(text)

            summary = summarize(text, mode)
            save_summary_pdf(summary)

            os.remove(file_path)

    return render_template("index.html", summary=summary)

    
@app.route("/download")
def download():
    return send_file("summary.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)