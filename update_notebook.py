import json

def update_cell(cells, func_name, new_code):
    for c in cells:
        if c.get("cell_type") == "code":
            source = "".join(c.get("source", []))
            if f"def {func_name}(" in source:
                c["source"] = [line + "\n" for line in new_code.strip().split("\n")]
                # remove the last newline from the last element to match standard jupyter format
                if c["source"]:
                    c["source"][-1] = c["source"][-1].rstrip("\n")
                return True
    return False

with open(r"d:\DL_PROJECT\DL_PROJECT.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new functions
clean_text_code = """import re

def clean_text(text):
    text = re.sub(r'\\[\\d+(,\\s*\\d+)*\\]', '', text)  # remove citations [1,2]
    text = re.sub(r'\\(.*?\\)', '', text)           # remove brackets (..)
    text = re.sub(r'http\\S+', '', text)           # remove URLs
    text = re.sub(r'\\s+', ' ', text)              # normalize spaces
    return text.strip()"""

adaptive_chunking_code = """from nltk.tokenize import sent_tokenize

def split_by_paragraph(text):
    paragraphs = text.split("\\n")
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

    return chunks"""

rank_chunks_code = """from sklearn.feature_extraction.text import TfidfVectorizer
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
    return ranked"""

summarize_chunks_code = """def get_summary_params(mode):
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

    return summaries"""

refine_summary_code = """def refine_summary(text, tokenizer, model, device, mode):
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

    return tokenizer.decode(ids[0], skip_special_tokens=True)"""

extract_key_points_code = """from nltk.tokenize import sent_tokenize
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
        return sentences[:num_points]"""

cells = nb.get("cells", [])
update_cell(cells, "clean_text", clean_text_code)
update_cell(cells, "adaptive_chunking", adaptive_chunking_code)
update_cell(cells, "rank_chunks", rank_chunks_code)
update_cell(cells, "summarize_chunks", summarize_chunks_code)
update_cell(cells, "refine_summary", refine_summary_code)
update_cell(cells, "extract_key_points", extract_key_points_code)

with open(r"d:\DL_PROJECT\DL_PROJECT.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully!")
