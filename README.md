# NLP Project 2 – Insurance Review Analysis (Streamlit App)

An interactive NLP workspace built with Streamlit that performs text cleaning,
topic modeling, supervised classification, deep learning, and information
retrieval on French insurance customer reviews.

---

## Project Structure

```
NLP_project/
├── streamlit_app.py        ← main application (single file)
├── requirements.txt        ← Python dependencies
├── README.md               ← this file
└── data/                   ← dataset folder (35 xlsx files)
    ├── avis_1_traduit.xlsx
    ├── avis_2_traduit.xlsx
    └── ... (up to avis_35_traduit.xlsx)
```

---

## How to Run (Step-by-Step)

### Prerequisites
- **Python 3.9 or higher** must be installed and available in your PATH.
  Check by running: `python --version`

---

### Step 1 – Open a terminal in the project folder

Navigate to the project folder in your terminal / command prompt:

```
cd path\to\NLP_project
```

---

### Step 2 – Create a virtual environment (recommended)

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3 – Install dependencies

```
pip install -r requirements.txt
```

> This installs: `streamlit`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`,
> `openpyxl`, `gensim`, `textblob`, `transformers`, `tensorflow`, `sentencepiece`.
>
> If any optional package fails to install (e.g. `tensorflow` on older hardware),
> the app still runs — those features are gracefully skipped.

---

### Step 4 – Launch the app

```
python -m streamlit run streamlit_app.py
```

The browser opens automatically at **http://localhost:8501**

If the browser does not open automatically, paste the URL manually.

---

## What Each Tab Does

| Tab | Content |
|-----|---------|
| **Overview** | Dataset statistics, sample cleaned rows, spelling-correction demo, download cleaned CSV/Excel |
| **Exploration & NLP tasks** | N-gram frequency charts, topic modeling (NMF), Word2Vec embeddings + t-SNE, zero-shot classification, auto-generated insurer summaries |
| **Supervised Models** | TF-IDF + Logistic Regression / Naïve Bayes for star rating, sentiment, and subject prediction; confusion matrices; error analysis; logistic explanation panel |
| **Insurer Analytics** | Per-insurer statistics, subject breakdown heatmap, semantic search over reviews, retrieval-based QA |
| **Deep Learning** | Keras embedding-layer text classifier (TensorFlow), training loss/accuracy curve, embedding-space visualization, TensorBoard export |

---
