
## Team Members:
## Name:Patel Vatsal Sureshbhai
## Student ID:202511040
## Name:Shivang Desai
## Student ID:202511034



# 📄 Adaptive PDF Summarization System using Deep Learning

---

## 🚀 Project Overview
This project aims to develop an end-to-end deep learning system capable of generating concise and meaningful summaries from long PDF documents. The system addresses the challenge of information overload by automatically extracting and condensing key insights from large textual data.

---

## 🎯 Objectives
- Develop a transformer-based summarization system using a pretrained model (**BART**)
- Fine-tune the model for abstractive text summarization
- Handle long PDF documents beyond model token limits
- Provide user-controlled summary lengths (short, medium, detailed)
- Build an end-to-end pipeline for PDF upload and summary generation

---

## 🧠 Key Concepts Used
- Transformer Architecture (Encoder-Decoder)
- Abstractive Text Summarization
- Transfer Learning & Fine-Tuning
- Tokenization & Sequence Modeling
- Hierarchical Summarization

---

## ⚙️ System Pipeline

```mermaid
graph TD
    A[PDF Upload] --> B[Text Extraction]
    B --> C[Adaptive Chunking]
    C --> D[Chunk-wise Summarization using BART]
    D --> E[Hierarchical Summarization]
    E --> F[Final Summary Output]
