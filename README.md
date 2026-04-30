# 📄 Smart PDF Summarizer using Hybrid Extractive-Abstractive Deep Learning

---

## 👥 Team Members
- **Patel Vatsal Sureshbhai** (202511040)  
- **Shivang Desai** (202511034)  

---

## 🚀 Objective & Description

The objective of this project is to build an intelligent system that can automatically generate structured summaries from PDF documents.

The application:
- Takes a PDF as input
- Extracts and cleans text
- Generates:
  - 📌 Title
  - 🔑 Key Points
  - 📝 Summary

The system uses a **Hybrid Extractive-Abstractive approach**, combining:
- Extractive methods (TF-IDF, NLTK)
- Abstractive methods (BART Transformer)

This ensures:
- Important information is not lost
- Output is readable and concise

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
    F --> L[Key Point Extraction]
