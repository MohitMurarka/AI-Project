
# 📚 PolarBrief AI - Legal Argument Analyzer

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MohitMurarka/AI-Project
cd polarbrief-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required Python packages include:

- `streamlit`
- `fpdf`
- `pdfplumber`
- `pytesseract`
- `pdf2image`
- `Pillow`
- `nltk`
- `scikit-learn`
- `langchain`
- `langchain-groq`
- `groq`
- `pydantic`
- `numpy`

### 🔐 3. Set Up Environment Variables (Windows CMD)

Before running the app, set your Groq API key in .env file:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Install Tesseract & Poppler

- **Windows:**
  - Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - Install [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

- **Linux/macOS:**

```bash
sudo apt install tesseract-ocr poppler-utils
```

Update `tesseract_path` and `poppler_path` accordingly in the .env file.

```
POPPLER_PATH=your_path
TESSERACT_PATH=your_path
```
---

## 🧪 Run the App

```bash
streamlit run app.py
```

---


**PolarBrief AI** is a Streamlit-based web application that allows users to upload legal PDF documents and uses advanced AI (Groq's LLaMA 3 via LangChain) to:

- Extract and summarize legal arguments.
- Detect polarity (Pro/Con).
- Score arguments based on relevance, clarity, and weight.
- Provide downloadable results in JSON and PDF format.
- Visualize top 5 pro/con arguments and citations directly in the UI.

---

## 🚀 Features

- 📄 Upload any legal PDF document.
- 🔍 Hybrid OCR (Tesseract) and native text extraction (pdfplumber).
- 🧠 AI-powered argument detection, heading generation, and summarization.
- ⚖️ Classify arguments as *Pro (Plaintiff)* or *Con (Defendant)* or *N/A(Neutral)*.
- 📈 Weighted scoring using LLM + TF-IDF centrality.
- 📥 Download ZIP bundle containing different json and pdf files .



## 📁 Output Files 

| File | Description |
|------|-------------|
| `all_arguments.json` | Full list of summary of analyzed paragraphs |
| `balanced_arguments.json` | Top 5 pro and 5 con legal arguments |
| `index.json` | Page, citation, and heading info only |
| `top_5_neutral.json` | Contains important argument with neutral polarity |
| `all_arguments.pdf` | Full PDF report |
| `index.pdf` | Index |
| `top_5_neutral.pdf`| Contains neutral arguments |
| `balanced_arguments.pdf`| Top 5 pro/con arguments |
| `Docs.zip` | All above in a single ZIP |

---

# ✅ PolarBrief Validation & Citation Accuracy Report

This document that summarizes the validation process and results for **PolarBrief AI - Legal Argument Analyzer**, which focuses on:

- 📌 Citation correctness (page/citation matching)
- ✅ Relevance and argument accuracy (manual validation)

---

## 📎 Attached Reports

- `validation_report.pdf`: Summary of citation and manual validation.
- `accuracy_report.ipynb`: Jupyter Notebook used for analysis (if needed for reproduction).

---

## 🧠 AI Model & Prompting

- Uses `llama3-8b-8192` via `LangChain` and `ChatGroq`.
- Prompts the LLM to:
  - Summarize in 75+ words without hallucination.
  - Generate a legal heading.
  - Classify polarity (Pro/Con).
  - Score the argument 0–100.

---
# doc-analyser
