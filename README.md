# NLP Project 2 – Insurance Customer Review Analysis

An interactive Natural Language Processing application built with **Streamlit** that
analyses **24,093 French insurance customer reviews** from **56 insurance companies**.
The app covers the full NLP pipeline — from raw data cleaning all the way to deep
learning and retrieval-based question answering.

---

## About the Dataset

| Property | Value |
|----------|-------|
| Source files | 35 Excel (.xlsx) files in the `data/` folder |
| Total reviews | 24,093 (after cleaning and deduplication) |
| Language | French (with English translations provided) |
| Insurance companies | 56 unique insurers |
| Star ratings | 1 to 5 stars |
| Columns used | `note`, `auteur`, `avis`, `assureur`, `produit`, `avis_en` |

---

## Features Implemented

| # | Feature | Method |
|---|---------|--------|
| 1 | Data cleaning & normalization | Regex, stop-word removal, accent stripping |
| 2 | Spell correction | TextBlob `.correct()` |
| 3 | N-gram frequency analysis | CountVectorizer |
| 4 | Topic modeling | NMF (Non-negative Matrix Factorization) |
| 5 | Word embeddings | Word2Vec (gensim) + LSA fallback |
| 6 | Embedding visualization | t-SNE 2D projection |
| 7 | Star rating prediction | TF-IDF + Logistic Regression / Naïve Bayes |
| 8 | Sentiment classification | positive / neutral / negative (~80% accuracy) |
| 9 | Subject classification | 6 categories (Pricing, Coverage, Claims…) |
| 10 | Prediction explanation | Logistic coefficient × TF-IDF term scores |
| 11 | Semantic search | TF-IDF cosine similarity |
| 12 | Question Answering | DistilBERT extractive QA + keyword fallback |
| 13 | Zero-shot classification | BART-large-MNLI (HuggingFace Transformers) |
| 14 | Insurer analytics | Pivot tables, averages, subject heatmap |
| 15 | Deep learning | Keras embedding model + TensorBoard export |

---

## Project Structure

```
NLP_project/
├── streamlit_app.py        ← entire application (single Python file)
├── requirements.txt        ← all Python dependencies
├── README.md               ← this file
└── data/                   ← dataset (35 Excel files)
    ├── avis_1_traduit.xlsx
    ├── avis_2_traduit.xlsx
    └── ... (up to avis_35_traduit.xlsx)
```

---

## How to Run — Option A: Clone from GitHub (Recommended)

```bash
git clone https://github.com/chanumolulalith/NLP_PROJECT.git
cd NLP_PROJECT
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**

---

## How to Run — Option B: From a downloaded folder

### Step 1 — Open a terminal inside the project folder

**Windows:**
```
cd C:\path\to\NLP_project
```

**macOS / Linux:**
```bash
cd /path/to/NLP_project
```

---

### Step 2 — Create a virtual environment (recommended)

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

### Step 3 — Install all dependencies

```
pip install -r requirements.txt
```

> **Note:** If `tensorflow` fails to install (requires Python 3.9 – 3.12),
> the app still runs completely — the Deep Learning tab will show a friendly
> message and all other 5 tabs work normally.

---

### Step 4 — Launch the app

```
python -m streamlit run streamlit_app.py
```

The browser opens automatically at **http://localhost:8501**
If it does not open, copy and paste the URL manually.

---

## App Tabs — What Each One Does

| Tab | What you can do |
|-----|----------------|
| **Overview** | See dataset statistics (24k reviews, 56 insurers), view sample cleaned rows, try spell correction, download the cleaned dataset as CSV or Excel |
| **Exploration & NLP tasks** | Generate n-gram frequency charts, run NMF topic modeling, train Word2Vec and explore the t-SNE word map, run zero-shot subject classification, generate insurer summaries |
| **Supervised Models** | Train Logistic Regression or Naïve Bayes for star rating / sentiment / subject prediction; view confusion matrix, classification report, and misclassified examples |
| **Prediction + Explanation** | Type any review and get instant predictions from all 3 models; see which words pushed the model toward or away from each prediction |
| **Insurer Analysis + Retrieval** | Filter by insurer and rating; view ranking table and subject heatmap; search reviews by meaning; ask natural-language questions over retrieved results |
| **Deep Learning** | Train a Keras embedding-layer model; view training loss curve; visualize the embedding space; export vectors for TensorBoard |

---

## Optional: TensorBoard

After training a deep learning model inside the app and clicking
**"Export embeddings for TensorBoard"**, run:

```
tensorboard --logdir "tensorboard_logs"
```

Then open **http://localhost:6006** in your browser.

---

## Python Version Compatibility

| Package | Requires | Status |
|---------|----------|--------|
| streamlit, pandas, scikit-learn | Python 3.8+ | Always works |
| textblob, transformers | Python 3.8+ | Always works |
| gensim | Python 3.8+ (needs C++ build tools) | sklearn fallback built-in |
| tensorflow | Python 3.9 – 3.12 only | Deep learning tab skipped on 3.13+ |

> The app was built and tested on **Python 3.14**. All features except
> TensorFlow work out of the box.

---

## GitHub Repository

[https://github.com/chanumolulalith/NLP_PROJECT](https://github.com/chanumolulalith/NLP_PROJECT)
