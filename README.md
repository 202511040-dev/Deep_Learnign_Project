# 📄 Smart PDF Summarizer using Hybrid Extractive-Abstractive Deep Learning

---

## 👥 Team Members
- Patel Vatsal Sureshbhai (202511040)  
- Shivang Desai (202511034)  

---

## 🚀 Objective & Description

The objective of this project is to build an intelligent system that can automatically generate structured summaries from PDF documents.

The application:
- Takes a PDF as input  
- Extracts and cleans text  
- Generates:
  - Title  
  - Key Points  
  - Summary  

The system uses a **Hybrid Extractive-Abstractive approach**, combining:
- Extractive methods (TF-IDF, NLTK)
- Abstractive methods (BART Transformer)

This ensures:
- Important information is preserved  
- Output is readable and concise  

---

## ⚙️ System Pipeline

PDF Upload → Text Extraction → Text Cleaning → Adaptive Chunking →  
TF-IDF Ranking → Select Important Chunks → BART Summarization →  
Combine Summaries → Refinement → Final Summary + Title  

Parallel: Key Point Extraction using NLTK

---

## 🧠 Methodology

### Extractive Phase
- Document is split into chunks  
- Important chunks selected using TF-IDF  

### Abstractive Phase
- Selected chunks summarized using BART  
- Combined and refined into final summary  

### Key Points
- Extracted using NLTK sentence scoring  
- Noise (tables, references) is removed  

---

## 📊 Results

| Metric   | Baseline | Improved |
|----------|---------|---------|
| ROUGE-1  | 0.3651  | 0.3932  |
| ROUGE-2  | 0.1605  | 0.1807  |
| ROUGE-L  | 0.2752  | 0.2884  |

### Observations
- Improved content coverage  
- Better readability and coherence  
- Reduced redundancy  

---

## 🖥️ Features

- Upload PDF  
- Select summary type (short / medium / detailed)  
- Generate title  
- Extract key points  
- Generate summary  
- Download summary as PDF  

---

## 🛠️ Installation & Running the Application

### 1. Clone the Repository
git clone https://github.com/202511040-dev/Deep_Learnign_Project/  
cd Deep_Learnign_Project

### 2. Create a Virtual Environment
python -m venv venv  

Activate it:

Windows:  
venv\Scripts\activate  

Mac/Linux:  
source venv/bin/activate  

### 3. Install Dependencies
pip install -r requirements.txt  

### 4. Run the Application
python app.py  

### 5. Open in Browser
http://127.0.0.1:5000/  

---

## ▶️ How to Use the Application

1. Open the application in your browser  
2. Upload a PDF file  
3. Select summary type:
   - Short → concise summary  
   - Medium → balanced summary  
   - Detailed → comprehensive summary  
4. Click **Generate Summary**  
5. View:
   - Title  
   - Key Points  
   - Summary  
6. Download summary as PDF  

---

## 📁 Project Structure

DL_PROJECT/  
│── app.py  
│── requirements.txt  
│── DL_PROJECT.ipynb
│── README.md  

---

## ⚠️ Limitations

- Depends on PDF text extraction quality  
- TF-IDF does not capture deep semantic meaning  
- Large models increase inference time  

---

## 🚀 Future Improvements

- Replace TF-IDF with embedding-based ranking  
- Use Longformer (LED) for long documents  
- Improve semantic understanding  
- Enhance UI/UX  

---

## 🌐 Deployment

The project is deployed using Hugging Face Spaces for real-time usage.

---

## 📌 Conclusion

This project demonstrates an effective hybrid summarization approach combining extractive and abstractive techniques. It generates structured and meaningful summaries from long PDF documents and can be extended for real-world applications.
