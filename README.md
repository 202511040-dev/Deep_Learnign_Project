# 📄 Smart PDF Summarizer using Hybrid Extractive-Abstractive Deep Learning

---

## 👥 Team Members
- **Patel Vatsal Sureshbhai** (202511040)  
- **Shivang Desai** (202511034)  

---

## 🚀 Project Overview
This project presents an end-to-end intelligent PDF summarization system that generates structured outputs including a **title, key points, and summary** from long documents.

The system uses a **hybrid extractive-abstractive pipeline**, combining traditional NLP techniques with deep learning models to ensure both **content relevance and readability**.

It is deployed as a web application where users can upload PDFs and receive summaries instantly.

---

## 🎯 Objectives
- Develop a robust PDF summarization system for long documents  
- Combine extractive and abstractive techniques for better performance  
- Handle noisy and unstructured PDF text  
- Support multiple summary lengths (short, medium, detailed)  
- Generate structured output (Title + Key Points + Summary)  
- Deploy the system using a user-friendly interface  

---

## 🧠 Key Concepts Used
- Transformer Architecture (BART)
- Abstractive Summarization
- Extractive Summarization (TF-IDF)
- Hierarchical Summarization
- Natural Language Processing (NLTK)
- Tokenization and Text Preprocessing
- PDF Text Extraction

---

## ⚙️ System Pipeline

```mermaid
graph TD
    A[PDF Upload] --> B[Text Extraction]
    B --> C[Text Cleaning]
    C --> D[Adaptive Chunking]
    D --> E[TF-IDF Ranking]
    E --> F[Select Important Chunks]
    F --> G[BART Summarization]
    G --> H[Combine Summaries]
    H --> I[Refinement Step]
    I --> J[Final Summary]
    I --> K[Title Generation]
    F --> L[Key Point Extraction (NLTK)]
