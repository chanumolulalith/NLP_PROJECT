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
NLP_PROJECT/
├── streamlit_app.py        ← entire application (single Python file)
├── requirements.txt        ← all Python dependencies
├── README.md               ← this file
└── data/                   ← dataset (35 Excel files)
    ├── avis_1_traduit.xlsx
    ├── avis_2_traduit.xlsx
    └── ... (up to avis_35_traduit.xlsx)
```

---

## How to Run — Option A: Using Git (Recommended)

> Use this if you have **Git installed** on your computer.

Open a terminal (Command Prompt on Windows, Terminal on Mac) and run these commands **one by one**:

```bash
# 1. Download the project (creates a folder called NLP_PROJECT)
git clone https://github.com/chanumolulalith/NLP_PROJECT.git

# 2. GO INSIDE the downloaded folder  ← very important, do not skip this!
cd NLP_PROJECT

# 3. Install all required packages
pip install -r requirements.txt

# 4. Start the app
python -m streamlit run streamlit_app.py
```

> **Common mistake:** Running `pip install -r requirements.txt` before doing `cd NLP_PROJECT`
> will give the error *"No such file or directory: requirements.txt"*.
> Always `cd NLP_PROJECT` first.

The app opens automatically at **http://localhost:8501**

---

### Quick copy-paste for Mac / Linux:

```bash
git clone https://github.com/chanumolulalith/NLP_PROJECT.git
cd NLP_PROJECT
pip3 install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

### Quick copy-paste for Windows:

```
git clone https://github.com/chanumolulalith/NLP_PROJECT.git
cd NLP_PROJECT
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

---

## How to Run — Option B: Download ZIP from GitHub (No Git needed)

> Use this if you do **not** have Git installed.

**Step 1 — Download the ZIP:**

1. Go to **https://github.com/chanumolulalith/NLP_PROJECT**
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Save it to your Desktop

**Step 2 — Extract the ZIP:**

- **Windows:** Right-click the ZIP → **"Extract All"** → Extract
- **macOS:** Double-click the ZIP — it extracts automatically into a folder called `NLP_PROJECT-main`

> The extracted folder is called **`NLP_PROJECT-main`** (GitHub adds `-main` to the name).

**Step 3 — Open a terminal INSIDE the extracted folder:**

- **Windows:** Open the `NLP_PROJECT-main` folder → click the address bar at the top → type `cmd` → press Enter
- **macOS:** Right-click the `NLP_PROJECT-main` folder → **"New Terminal at Folder"**

**Step 4 — Confirm you are in the right folder** (you should see `requirements.txt` listed):

```bash
# Windows
dir

# macOS / Linux
ls
```

You should see: `streamlit_app.py`, `requirements.txt`, `README.md`, `data/`

**Step 5 — Install dependencies:**

```bash
# Windows
pip install -r requirements.txt

# macOS / Linux
pip3 install -r requirements.txt
```

**Step 6 — Launch the app:**

```bash
# Windows
python -m streamlit run streamlit_app.py

# macOS / Linux
python3 -m streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**
If it does not open, copy and paste the URL manually into your browser.

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
| gensim | Python 3.8+ (needs C++ build tools) | sklearn fallback built-in — not required |
| tensorflow | Python 3.9 – 3.12 only | Deep learning tab skipped on Python 3.13+ |

> The app was built and tested on **Python 3.14**. All features except
> TensorFlow work out of the box. `pip install -r requirements.txt` installs
> only compatible packages and completes without errors on any Python version.

---

## GitHub Repository

[https://github.com/chanumolulalith/NLP_PROJECT](https://github.com/chanumolulalith/NLP_PROJECT)
