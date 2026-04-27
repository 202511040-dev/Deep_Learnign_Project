import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# --- CELL ---

from dataclasses import dataclass

@dataclass
class Config:
    MODEL_NAME: str = "facebook/bart-large-cnn"
    MAX_INPUT_LENGTH: int = 1024
    MAX_SUMMARY_LENGTH: int = 150
    MIN_SUMMARY_LENGTH: int = 40

config = Config()

# --- CELL ---

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained(config.MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(config.MODEL_NAME).to(device)

# --- CELL ---

def summarize_text(text: str):
    inputs = tokenizer(
        text,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config.MAX_SUMMARY_LENGTH,
            min_length=config.MIN_SUMMARY_LENGTH,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- CELL ---

text = """
Artificial Intelligence is transforming industries by enabling machines
to learn from data, automate tasks, and provide intelligent insights.
Deep learning models, especially transformers, have revolutionized
natural language processing tasks such as summarization and translation.
"""

summary = summarize_text(text)

print("Original:\n", text)
print("\nSummary:\n", summary)

# --- CELL ---

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")

# Reduce dataset size (VERY IMPORTANT for Colab)
train_data = dataset["train"].select(range(8000))
val_data = dataset["validation"].select(range(1000))

print(train_data[0])

# --- CELL ---

from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# --- CELL ---

max_input_length = 1024
max_target_length = 150

def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- CELL ---

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)

# --- CELL ---

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# --- CELL ---

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=100,
    fp16=True,
    push_to_hub=False
)

# --- CELL ---

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- CELL ---

trainer.train()

# --- CELL ---

from google.colab import drive
drive.mount('/content/drive')

# --- CELL ---

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

MODEL_PATH = "/content/drive/MyDrive/DL_PROJECT/pdf_summarizer_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

print("✅ Model Loaded Successfully")

# --- CELL ---

# import os

# save_path = "/content/drive/MyDrive/DL_PROJECT/pdf_summarizer_model"

# # Create folder if it doesn't exist
# os.makedirs(save_path, exist_ok=True)

# --- CELL ---

# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

# --- CELL ---

# print(os.listdir(save_path))

# --- CELL ---

def baseline_summary(text, tokenizer, model, device):
    """
    Direct summarization (NO chunking)
    Used as baseline for comparison
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# --- CELL ---

!pip install -q pdfplumber nltk

# --- CELL ---

import pdfplumber
import nltk
import torch

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# --- CELL ---

def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text

# --- CELL ---

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# --- CELL ---

import re

def clean_text(text):
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)  # remove citations [1,2]
    text = re.sub(r'\(.*?\)', '', text)           # remove brackets (..)
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'\s+', ' ', text)              # normalize spaces
    return text.strip()

# --- CELL ---

def split_by_paragraph(text):
    paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]

# --- CELL ---

def get_dynamic_max_length(text):
    length = len(text.split())

    if length < 2000:
        return 800
    elif length < 5000:
        return 900
    else:
        return 1000

# --- CELL ---

from nltk.tokenize import sent_tokenize

def split_by_paragraph(text):
    paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]

def get_dynamic_max_length(text):
    length = len(text.split())
    if length < 2000: return 800
    elif length < 5000: return 900
    else: return 1000

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
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --- CELL ---

from transformers import BartForConditionalGeneration, BartTokenizer

model_path = "/content/drive/MyDrive/DL_PROJECT/pdf_summarizer_model"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to("cuda")

# --- CELL ---

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def rank_chunks(chunks):
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    scores = np.sum(X.toarray(), axis=1)
    
    normalized_scores = []
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        normalized_score = scores[i] / (word_count + 1e-5)
        normalized_scores.append(normalized_score)
        
    ranked = [c for _, c in sorted(zip(normalized_scores, chunks), reverse=True)]
    return ranked

# --- CELL ---

def select_chunks(ranked_chunks, mode):
    """
    Select top chunks based on user mode
    """

    if mode == "short":
        return ranked_chunks[:3]
    elif mode == "detailed":
        return ranked_chunks[:12]
    else:  # medium
        return ranked_chunks[:6]

# --- CELL ---

def get_summary_params(mode="medium"):
    if mode == "short":
        return {
            "max_length": 80,
            "min_length": 30,
            "num_beams": 4
        }
    elif mode == "detailed":
        return {
            "max_length": 250,
            "min_length": 100,
            "num_beams": 5
        }
    else:  # medium
        return {
            "max_length": 150,
            "min_length": 60,
            "num_beams": 4
        }

# --- CELL ---

def get_summary_params(mode):
    if mode == "short": return {"max_length": 80, "min_length": 30, "num_beams": 4}
    elif mode == "detailed": return {"max_length": 250, "min_length": 100, "num_beams": 5}
    else: return {"max_length": 150, "min_length": 60, "num_beams": 4}

def summarize_chunks(chunks, mode="medium"):
    summaries = []
    params = get_summary_params(mode)

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)

        ids = model.generate(
            inputs["input_ids"],
            max_length=params["max_length"],
            min_length=params["min_length"],
            num_beams=params["num_beams"],
            length_penalty=2.0
        )

        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    return summaries

# --- CELL ---

def refine_summary(text, tokenizer, model, device, mode):
    if mode == "short": max_len, min_len, beams = 80, 30, 4
    elif mode == "detailed": max_len, min_len, beams = 300, 120, 6
    else: max_len, min_len, beams = 150, 60, 5

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=beams,
        length_penalty=2.2,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# --- CELL ---

def remove_redundancy(text):
    sentences = text.split(".")

    seen = set()
    unique_sentences = []

    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    return ". ".join(unique_sentences)

# --- CELL ---

def generate_title(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    ids = model.generate(
        inputs["input_ids"],
        max_length=20,
        num_beams=4
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# --- CELL ---

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_key_points(text, num_points=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_points:
        return sentences

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(sentences)
        scores = np.sum(X.toarray(), axis=1)
        ranked = [s for _, s in sorted(zip(scores, sentences), reverse=True)]
        return ranked[:num_points]
    except ValueError:
        return sentences[:num_points]

# --- CELL ---

def hierarchical_summarization(text, mode="medium"):

    # Step 1: Chunking
    chunks = adaptive_chunking(text, tokenizer)
    print(f"Total chunks before ranking: {len(chunks)}")

    # Step 2: Ranking
    ranked_chunks = rank_chunks(chunks)

    # Step 3: Selection
    selected_chunks = select_chunks(ranked_chunks, mode)
    print(f"Selected chunks: {len(selected_chunks)}")

    # Step 4: Chunk Summarization
    chunk_summaries = summarize_chunks(selected_chunks, mode)

    # Step 5: Combine summaries
    combined_summary = " ".join(chunk_summaries)
    combined_summary = remove_redundancy(combined_summary)

    combined_summary = combined_summary.replace("  ", " ")

    # Step 6: Final Refinement (ONLY ONCE)
    final_summary = refine_summary(combined_summary, tokenizer, model, device, mode)
    final_summary = remove_redundancy(final_summary)

    title = generate_title(final_summary, tokenizer, model, device)
    points = extract_key_points(final_summary)

    return {
        "title": title,
        "summary": final_summary,
        "points": points
    }

    return final_summary

# --- CELL ---

mode = input("Choose summary type (short / medium / detailed): ").strip().lower()

if mode not in ["short", "medium", "detailed"]:
    mode = "medium"

# --- CELL ---

pdf_path ="/content/drive/MyDrive/SK_Papers/Drive4GPT.pdf"  # upload your PDF


text = extract_text_from_pdf(pdf_path)
text = clean_text(text)

print("🔵 Running BASELINE summarization...\n")

baseline_output = baseline_summary(text, tokenizer, model, device)

print("Baseline Summary:\n")
print(baseline_output)

print("Chunk based  Summary:\n")
final_summary = hierarchical_summarization(text, mode)

print("\nFinal Summary:\n")
print(final_summary)

# --- CELL ---

!pip install evaluate

# --- CELL ---

import evaluate
!pip install rouge_score

rouge = evaluate.load("rouge")

# --- CELL ---

def evaluate_summaries(reference, baseline, improved):

    results_baseline = rouge.compute(
        predictions=[baseline],
        references=[reference]
    )

    results_improved = rouge.compute(
        predictions=[improved],
        references=[reference]
    )

    print("🔵 BASELINE ROUGE:")
    print(results_baseline)

    print("\n🟢 IMPROVED MODEL ROUGE:")
    print(results_improved)

# --- CELL ---

reference_summary = "Write or take actual summary here"

evaluate_summaries(
    reference_summary,
    baseline_output,
    final_summary
)

# --- CELL ---

print("\n🔵 BASELINE SUMMARY:\n", baseline_output)
print("\n🟢 IMPROVED SUMMARY:\n", final_summary)

# --- CELL ---

