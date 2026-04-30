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


---

# 🧠 Methodology

### 🔹 Extractive Phase
- Document is split into chunks  
- Important chunks selected using TF-IDF  

### 🔹 Abstractive Phase
- Selected chunks summarized using BART  
- Combined and refined into final summary  

### 🔹 Key Points Extraction
- Extracted using NLTK sentence scoring  
- Noise (tables, references) is removed  

---

## 📊 Results

| Metric   | Baseline | Improved |
|----------|---------|---------|
| ROUGE-1  | 0.3651  | 0.3932  |
| ROUGE-2  | 0.1605  | 0.1807  |
| ROUGE-L  | 0.2752  | 0.2884  |

### ✅ Observations
- Improved content coverage  
- Better readability and coherence  
- Reduced redundancy  

---

## 🖥️ Features

- 📂 Upload PDF  
- 📊 Select summary type (short / medium / detailed)  
- 📌 Generate Title  
- 🔑 Extract Key Points  
- 📝 Generate Summary  
- 📥 Download summary as PDF  

---

## 🛠️ Installation & Running the Application

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
