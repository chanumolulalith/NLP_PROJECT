"""
NLP Project 2 – Insurance Customer Reviews Analysis
=====================================================
This Streamlit application performs end-to-end Natural Language Processing on
French insurance customer reviews collected from multiple insurers.

Pipeline overview:
  1. Data loading  – reads all *.xlsx files from the data/ folder
  2. Text cleaning – normalisation, stop-word removal, optional accent-stripping
                     and bi-gram augmentation
  3. Exploration   – n-gram frequency charts, topic modeling (NMF), Word2Vec
                     embeddings with t-SNE visualisation
  4. Supervised    – TF-IDF + Logistic Regression / Naïve Bayes for three tasks:
                       (a) star-rating prediction  (1–5)
                       (b) sentiment classification (positive / neutral / negative)
                       (c) subject-category tagging (Pricing, Coverage, …)
  5. Prediction    – interactive inference with logistic explanation (influential terms)
  6. Insurer intel – per-insurer statistics, semantic search, retrieval-based QA
  7. Deep learning – Keras embedding model, TensorBoard projector export (optional)

Author: NLP Project Group
"""
from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import io
from pathlib import Path
import unicodedata

# ── Third-party: visualisation and data manipulation ──────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# ── Streamlit UI framework ────────────────────────────────────────────────
import streamlit as st

# ── Scikit-learn: unsupervised, supervised, and evaluation utilities ───────
from sklearn.decomposition import NMF           # topic modelling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE               # 2-D word-vector projection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity  # semantic search
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# ── Page configuration (must be the first Streamlit call) ─────────────────
st.set_page_config(
    page_title="NLP Supervised Learning Project",
    page_icon="🧠",
    layout="wide",
)

# ── Path configuration ─────────────────────────────────────────────────────
# We support two possible data layouts:
#   1. data/Traduction avis clients/*.xlsx  (original assignment folder)
#   2. data/*.xlsx                          (flat layout used in this submission)
BASE_DIR = Path(__file__).resolve().parent
_DATA_SUBDIR = BASE_DIR / "data" / "Traduction avis clients"
_DATA_ROOT = BASE_DIR / "data"
DATA_DIR = _DATA_SUBDIR if _DATA_SUBDIR.exists() else _DATA_ROOT

# Directory where TensorBoard projector files will be written
TB_LOG_DIR = BASE_DIR / "tensorboard_logs"

# Name of the column that contains the final pre-processed text fed to models
TEXT_COLUMN = "text_for_model"

# ── Stop-word list ────────────────────────────────────────────────────────
# Bilingual (French + English) stop words removed during tokenisation.
# These high-frequency function words carry no discriminative information
# and would dominate TF-IDF and n-gram counts if left in place.
STOP_WORDS = {
    "a",
    "à",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "ceux",
    "cette",
    "cet",
    "celles",
    "celui",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "dun",
    "elle",
    "elles",
    "en",
    "et",
    "est",
    "eu",
    "eux",
    "fait",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "ma",
    "mais",
    "me",
    "moi",
    "mon",
    "ne",
    "nous",
    "on",
    "ou",
    "pas",
    "par",
    "pas",
    "pour",
    "plus",
    "que",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "son",
    "sur",
    "t",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vous",
    "y",
    "the",
    "and",
    "are",
    "is",
    "in",
    "it",
    "of",
    "that",
    "to",
    "was",
    "with",
    "for",
    "as",
    "have",
    "at",
    "not",
    "be",
    "this",
    "by",
    "or",
    "from",
    "so",
    "if",
    "who",
    "its",
    "just",
    "also",
    "can",
    "will",
    "my",
    "your",
    "me",
    "our",
}

# ── Rule-based subject keyword dictionary ─────────────────────────────────
# Maps each insurance topic category to a list of French/English trigger words.
# Used by infer_subject() to assign a subject label to each review when no
# supervised subject model has been trained yet.  This forms the training
# signal for the supervised subject classifier in the Supervised Models tab.
SUBJECT_KEYWORDS = {
    "Pricing": [
        "prix",
        "tarif",
        "prime",
        "coût",
        "cout",
        "élevé",
        "trop",
        "pas cher",
        "cher",
        "payment",
        "cost",
        "premium",
        "expensive",
        "cheap",
        "discount",
        "tarifs",
        "augment",
    ],
    "Coverage": [
        "garantie",
        "garant",
        "couverture",
        "assurance",
        "risque",
        "franchise",
        "souscription",
        "protection",
        "indemnisation",
        "reprise",
        "rembourse",
        "coverage",
        "coverage",
        "policy",
        "claim",
    ],
    "Enrollment": [
        "inscription",
        "souscrire",
        "contrat",
        "adhérer",
        "adhésion",
        "signer",
        "open",
        "join",
        "register",
        "onboarding",
        "account",
        "compte",
    ],
    "Customer Service": [
        "service",
        "support",
        "conseiller",
        "agent",
        "réclamation",
        "accueil",
        "téléphone",
        "call",
        "respons",
        "aide",
        "attente",
        "personnel",
        "agent",
        "rude",
        "sympathique",
        "polite",
        "service",
        "client",
    ],
    "Claims Processing": [
        "sinistre",
        "expert",
        "expertise",
        "garage",
        "indemn",
        "réparation",
        "dossier",
        "constat",
        "dégât",
        "claim",
        "claim",
        "claims",
        "assistance",
        "perte",
        "dommage",
        "règlement",
        "remboursement",
    ],
    "Cancellation": [
        "résiliation",
        "resilie",
        "annuler",
        "cancel",
        "suppres",
        "résilié",
        "résilié",
        "résilier",
        "break",
        "clôture",
        "fin contrat",
        "abandonner",
    ],
}


# ── Optional-library helpers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def textblob_available() -> bool:
    """Return True if TextBlob is installed, False otherwise.

    Cached so the import attempt happens only once per session.
    All spell-correction and translation helpers call this guard first
    to avoid ImportError at runtime.
    """
    try:
        from textblob import TextBlob

        return True
    except Exception:
        return False


def simple_textblob_correction(text: Any) -> str:
    """Apply TextBlob spell correction to a single text string.

    TextBlob's .correct() uses a word-frequency English dictionary.
    It works best on English text; French correction is approximate.
    Falls back silently to the original string if TextBlob is unavailable.
    """
    if not text or pd.isna(text):
        return ""
    raw = str(text).strip()
    if not raw or not textblob_available():
        return raw
    try:
        from textblob import TextBlob

        return str(TextBlob(raw).correct())
    except Exception:
        return raw


def text_to_translation(text: Any, target: str = "en") -> str:
    """Translate *text* into *target* language using TextBlob's Google Translate wrapper.

    Requires an internet connection on first call.
    Falls back to the original string when TextBlob or the network is unavailable.
    Limited to 500 rows per export to keep response times reasonable.
    """
    if not text or pd.isna(text):
        return ""
    raw = str(text).strip()
    if not raw or not textblob_available():
        return raw
    try:
        from textblob import TextBlob

        translated = TextBlob(raw).translate(to=target)
        return str(translated)
    except Exception:
        return raw


def build_delivery_dataset(
    df: pd.DataFrame,
    apply_spell_check: bool = False,
    apply_translation: bool = False,
) -> pd.DataFrame:
    """Build the multi-column submission-ready dataset.

    Steps:
      1. Apply basic text cleaning to both French and English review columns.
      2. (Optional) Fill missing translations using TextBlob (max 500 rows each direction).
      3. (Optional) Add spell-corrected variants of both language columns.
      4. Select and order a fixed set of columns required by the assignment deliverable.

    Parameters
    ----------
    df               : the full loaded dataset
    apply_spell_check: whether to compute TextBlob-corrected text columns
    apply_translation: whether to fill missing EN→FR or FR→EN translations
    """
    base = df.copy()
    # Produce a clean version of the raw French and English review text
    base = base.assign(
        text_fr_clean=base["raw_review_fr"].map(clean_text) if "raw_review_fr" in base else "",
        text_en_clean=base["raw_review_en"].map(clean_text) if "raw_review_en" in base else "",
    )
    # Pre-populate translation columns from existing values in the dataset
    base["translation_source_fr_to_en"] = base["avis_en"].fillna("")
    base["translation_source_en_to_fr"] = base["raw_review_fr"].fillna("")
    if apply_translation and ("raw_review_fr" in base.columns):
        # Only translate rows that still have an empty English translation
        need_en = base["translation_source_fr_to_en"].str.strip().eq("")
        en_candidates = base.loc[need_en, "raw_review_fr"].head(500).index
        for idx in en_candidates:
            base.loc[idx, "translation_source_fr_to_en"] = text_to_translation(
                base.loc[idx, "raw_review_fr"], target="en"
            )
    if apply_translation and ("raw_review_en" in base.columns):
        # Translate reviews that have English text but no French translation
        need_fr = base["translation_source_en_to_fr"].str.strip().eq("")
        fr_candidates = base.loc[need_fr, "raw_review_en"].head(500).index
        for idx in fr_candidates:
            base.loc[idx, "translation_source_en_to_fr"] = text_to_translation(
                base.loc[idx, "raw_review_en"], target="fr"
            )

    if apply_spell_check:
        # Run TextBlob spell correction on both language columns
        base["text_fr_spell_corrected"] = base["raw_review_fr"].map(simple_textblob_correction)
        base["text_en_spell_corrected"] = base["raw_review_en"].map(simple_textblob_correction)
    else:
        # Default: spell-corrected columns are identical to the basic cleaned versions
        base["text_fr_spell_corrected"] = base["text_fr_clean"]
        base["text_en_spell_corrected"] = base["text_en_clean"]

    selected = [
        "note",
        "sentiment",
        "subject",
        "assureur",
        "produit",
        "type",
        "reviewer",
        "date_publication",
        "date_exp",
        "raw_review_fr",
        "raw_review_en",
        "text_fr_clean",
        "text_en_clean",
        "text_fr_spell_corrected",
        "text_en_spell_corrected",
        "translation_source_fr_to_en",
        "translation_source_en_to_fr",
        "source_file",
        "word_count",
        "char_count",
        "text_for_model",
    ]
    missing = [col for col in selected if col not in base.columns]
    for col in missing:
        base[col] = ""
    return base[selected].copy()


# ── Text normalisation and cleaning functions ─────────────────────────────

def normalize_text(text: Any) -> str:
    """Lowercase and strip noise from a raw review string.

    Operations applied in order:
      - Convert to lowercase and replace non-breaking spaces
      - Remove URLs and HTML tags
      - Collapse whitespace characters (\\r \\n \\t)
      - Keep only French/Latin letters, digits, spaces, and apostrophes
      - Remove all digit sequences (numbers carry little NLP signal here)
      - Collapse multiple spaces into one
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().replace("\u00a0", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Retain French accented characters in addition to basic ASCII letters
    text = re.sub(r"[^a-zàâçéèêëîïôöùûüÿñæœ0-9\s']", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    """Remove diacritics from text using Unicode NFKD decomposition.

    Used in the advanced preprocessing mode when the user enables
    'Strip accents' in the sidebar.  This maps e.g. 'é' → 'e', 'ç' → 'c',
    which can improve generalisation for bag-of-words models on French text.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    # Category 'Mn' = Mark, Nonspacing (i.e., diacritic combining characters)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def clean_text_advanced(
    text: Any,
    strip_accents_from_text: bool = False,
    add_bigrams: bool = False,
    min_token_length: int = 2,
) -> str:
    """Advanced text cleaning with optional accent removal and bi-gram injection.

    After basic normalisation the function:
      1. Optionally strips accents (e.g. 'élevé' → 'eleve').
      2. Removes stop words and short tokens.
      3. Optionally appends contiguous bi-gram tokens (e.g. 'service_client')
         to the unigram list, giving the downstream TF-IDF vectoriser access
         to important two-word phrases without a separate n-gram parameter.
    """
    if pd.isna(text):
        return ""
    base = normalize_text(text)
    if strip_accents_from_text:
        base = strip_accents(base)
    tokens = [t for t in base.split(" ") if t and t not in STOP_WORDS and len(t) >= min_token_length]
    if add_bigrams and len(tokens) >= 2:
        # Create underscore-joined bi-grams and append them to the token list
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        tokens.extend(bigrams)
    return " ".join(tokens)


def build_preprocessed_dataset(
    df: pd.DataFrame,
    use_advanced: bool = False,
    strip_accents_from_text: bool = False,
    add_bigrams: bool = False,
    min_token_length: int = 2,
) -> pd.DataFrame:
    """Re-clean the TEXT_COLUMN in *df* using advanced settings if requested.

    When *use_advanced* is False the dataset is returned unchanged (basic cleaning
    was already applied during load_dataset).  Otherwise each text is re-processed
    with clean_text_advanced and rows that become empty are dropped.
    This preprocessed copy is used by all tabs (models, topics, Word2Vec, etc.).
    """
    if not use_advanced:
        return df
    processed = df.copy()
    processed[TEXT_COLUMN] = processed[TEXT_COLUMN].map(
        lambda t: clean_text_advanced(
            t,
            strip_accents_from_text=strip_accents_from_text,
            add_bigrams=add_bigrams,
            min_token_length=min_token_length,
        )
    )
    # Drop rows where the text became empty after aggressive cleaning
    processed = processed[processed[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)
    return processed


def tokenize_text(text: str) -> List[str]:
    """Normalise *text* and return a list of meaningful tokens.

    Applies normalize_text then filters out stop words and tokens shorter
    than 2 characters.  Used by Word2Vec training and the summary generator.
    """
    base = normalize_text(text)
    return [t for t in base.split(" ") if t and t not in STOP_WORDS and len(t) > 1]


def clean_text(text: Any) -> str:
    """Return a single cleaned string from *text* (join of tokenize_text output).

    This is the default cleaning function applied to every review during
    data loading.  The result is stored in 'text_fr_clean' / 'text_en_clean'.
    """
    return " ".join(tokenize_text(text))


def to_sentiment(rating: Any) -> str:
    """Convert a numeric star rating (1–5) to a three-class sentiment label.

    Mapping:
      4–5  → 'positive'
      3    → 'neutral'
      1–2  → 'negative'

    This label is used as the target for the binary/ternary sentiment classifier.
    """
    if pd.isna(rating):
        return "neutral"
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def infer_subject(clean_text_value: str) -> str:
    """Assign a subject category to a review using keyword matching.

    Iterates over SUBJECT_KEYWORDS and returns the first matching category.
    This rule-based approach creates the 'subject' column that is later used
    as training labels for the supervised subject classifier.
    Returns 'Other' when no keyword matches.
    """
    t = clean_text_value.lower()
    for subject, keywords in SUBJECT_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return subject
    return "Other"


# ── Data loading ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load, concatenate, and clean all .xlsx review files from DATA_DIR.

    Processing steps:
      1. Read every *.xlsx file found in DATA_DIR and tag each row with its
         source filename.
      2. Enforce required columns (fill missing ones with pd.NA).
      3. Coerce 'note' to numeric, drop rows outside [1, 5], round to integer.
      4. Deduplicate on (author, review text, publication date).
      5. Parse date columns with dayfirst=True (French date format DD/MM/YYYY).
      6. Produce clean text columns: text_fr_clean, text_en_clean.
      7. Select text_for_model = French clean text when available, else English.
      8. Derive word_count, char_count, sentiment label, and subject label.

    The result is cached by Streamlit so the heavy file I/O and cleaning
    runs only once per session.
    """
    if not DATA_DIR.exists():
        return pd.DataFrame()

    files = sorted(DATA_DIR.glob("*.xlsx"))
    if not files:
        return pd.DataFrame()

    frames = []
    for file in files:
        try:
            df = pd.read_excel(file)
            df["source_file"] = file.name
            frames.append(df)
        except Exception as exc:
            st.warning(f"Could not read {file.name}: {exc}")
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Columns expected in every file; create empty columns for any that are missing
    required = [
        "note",
        "auteur",
        "avis",
        "assureur",
        "produit",
        "type",
        "date_publication",
        "date_exp",
        "avis_en",       # English translation provided in the dataset
        "avis_cor",      # corrected French review
        "avis_cor_en",   # corrected English review
    ]
    for col in required:
        if col not in data.columns:
            data[col] = pd.NA

    data = data[required + ["source_file"]].copy()

    # Force star ratings to numeric and keep only valid 1–5 values
    data["note"] = pd.to_numeric(data["note"], errors="coerce")
    data = data.dropna(subset=["note"]).copy()
    data = data[(data["note"] >= 1) & (data["note"] <= 5)]
    data["note"] = data["note"].round().astype(int)

    # Remove exact duplicate reviews from the same author on the same date
    data = data.drop_duplicates(subset=["auteur", "avis", "date_publication"]).reset_index(drop=True)

    # Parse dates using French format (day first)
    for dcol in ["date_publication", "date_exp"]:
        data[dcol] = pd.to_datetime(data[dcol], errors="coerce", dayfirst=True)

    # Convenience aliases used throughout the app
    data["reviewer"] = data["auteur"].fillna("anonymous")
    data["raw_review_fr"] = data["avis"].fillna("")
    data["raw_review_en"] = data["avis_en"].fillna("")

    # Produce cleaned versions of both language columns
    data["text_fr_clean"] = data["raw_review_fr"].map(clean_text)
    data["text_en_clean"] = data["raw_review_en"].map(clean_text)

    # Keep track of the original translation mapping for the export file
    data["translation_source_fr_to_en"] = data["raw_review_en"]
    data["translation_source_en_to_fr"] = data["raw_review_fr"]

    # Prefer French text; fall back to English when French is empty
    data["has_french_text"] = data["text_fr_clean"].str.len() > 0
    data["text_for_model"] = np.where(
        data["has_french_text"],
        data["text_fr_clean"],
        data["text_en_clean"],
    )

    # Compute simple corpus statistics used in the Overview tab
    data["word_count"] = data["text_for_model"].str.split().str.len().fillna(0).astype(int)
    data["char_count"] = data["text_for_model"].str.len().fillna(0).astype(int)

    # Derive classification target labels from the star rating
    data["sentiment"] = data["note"].map(to_sentiment)
    # Assign a topic subject using keyword matching (training signal for supervised model)
    data["subject"] = data["text_for_model"].map(infer_subject)

    return data


# ── Search index and n-gram analysis ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_search_index(df: pd.DataFrame, text_column: str = TEXT_COLUMN):
    """Build a TF-IDF document matrix used for cosine-similarity search.

    We use unigrams + bigrams (ngram_range=(1,2)) and limit the vocabulary
    to 25,000 features to keep memory usage reasonable on a large corpus.
    The returned (vectorizer, matrix) pair is reused by semantic_search().

    @st.cache_resource ensures the matrix is built once and shared across
    all user interactions in the session.
    """
    corpus = df[text_column].fillna("")
    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=25000,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def top_ngrams(
    corpus: pd.Series,
    ngram_range: Tuple[int, int] = (1, 1),
    top_k: int = 40,
) -> pd.DataFrame:
    """Compute the most frequent n-grams in *corpus*.

    Uses CountVectorizer (raw term counts, not TF-IDF) so that the bar chart
    reflects actual frequency rather than document-relative importance.

    Parameters
    ----------
    corpus     : Series of pre-cleaned review texts
    ngram_range: (min_n, max_n) – e.g. (2, 2) for pure bigrams
    top_k      : number of top terms to return
    """
    docs = corpus.fillna("").astype(str)
    vectorizer = CountVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["term", "count"])
    # Sum across all documents to get corpus-level term frequencies
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    idx = np.argsort(counts)[::-1][:top_k]
    out = pd.DataFrame({"term": terms[idx], "count": counts[idx].astype(int)})
    return out


def plot_top_terms(df_terms: pd.DataFrame, title: str):
    """Render a horizontal bar chart of the most frequent terms in the app."""
    if df_terms.empty:
        st.info("No frequent terms found. Increase data size or reduce n-gram minimums.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df_terms["term"][::-1], df_terms["count"][::-1], color="#4b7bec")
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.grid(alpha=0.2, axis="x")
    st.pyplot(fig)


# ── Topic modelling (NMF) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Running topic model...")
def run_topic_model(
    df: pd.DataFrame,
    n_topics: int = 6,
    top_words: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    text_column: str = TEXT_COLUMN,
):
    """Discover latent topics in the review corpus using Non-negative Matrix Factorisation.

    Method:
      1. Vectorise the cleaned texts with TF-IDF (unigrams + bigrams by default).
      2. Factorise the TF-IDF matrix X ≈ W · H using NMF:
           W (n_docs × n_topics)  – document–topic weight matrix
           H (n_topics × n_terms) – topic–term weight matrix
      3. For each topic row in H, pick the top *top_words* highest-weight terms.

    NMF with 'nndsvd' initialisation converges faster than random init and
    produces more stable topics across runs.

    Returns a dict with:
      'topics'        : DataFrame of topic index + top terms + weight sum
      'document_topic': W matrix (document assignments, not shown in UI currently)
    """
    corpus = df[text_column].dropna().astype(str)
    if corpus.str.len().lt(5).all():
        return None

    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.9,
        max_features=12000,
    )
    X = vectorizer.fit_transform(corpus)
    if X.shape[0] < 4 or X.shape[1] < 2:
        return None

    # Cap n_topics so NMF never requests more components than the matrix rank
    max_allowed = min(n_topics, X.shape[0] - 1, X.shape[1] - 1)
    if max_allowed < 2:
        return None
    model = NMF(
        n_components=max_allowed,
        random_state=42,
        max_iter=300,
        init="nndsvd",   # deterministic initialisation for reproducibility
    )
    W = model.fit_transform(X)
    H = model.components_   # shape: (n_topics, n_terms)
    feature_names = np.array(vectorizer.get_feature_names_out())

    topic_list: List[Dict[str, Any]] = []
    for tidx, topic in enumerate(H):
        # Sort term weights descending and pick the top-*top_words* terms
        top_idx = np.argsort(topic)[::-1][:top_words]
        terms = [feature_names[i] for i in top_idx]
        scores = topic[top_idx]
        topic_list.append(
            {
                "topic": tidx + 1,
                "top_terms": ", ".join(terms),
                "weight_sum": float(np.sum(scores)),
            }
        )
    topics = pd.DataFrame(topic_list)
    return {"topics": topics, "document_topic": W}


# ── Word embeddings: gensim Word2Vec + pure-sklearn fallback ──────────────
#
# We always try gensim first (true skip-gram Word2Vec).
# If gensim is not installed or fails to compile on the current Python version,
# we fall back to LSA (TF-IDF + TruncatedSVD) which produces comparable word
# vectors using only standard scikit-learn.  Both backends expose the same
# .wv.key_to_index / .wv[word] / .wv.most_similar() interface so the rest
# of the code never needs to know which backend was used.

class _SimpleWordVectors:
    """Pure-numpy word vector store that mimics the gensim KeyedVectors API.

    Stores L2-normalised word vectors and supports:
      - key_to_index : dict mapping word → integer index
      - __getitem__  : retrieve the raw (un-normalised) vector for a word
      - most_similar : return the top-N cosine-nearest neighbours

    Used as the .wv attribute of _SklearnWord2VecModel.
    """

    def __init__(self, word_vecs: Dict[str, np.ndarray]) -> None:
        self.key_to_index: Dict[str, int] = {w: i for i, w in enumerate(word_vecs)}
        self._words: List[str] = list(word_vecs)
        self._matrix: np.ndarray = np.stack(list(word_vecs.values()))
        # Pre-compute a row-normalised matrix for fast cosine similarity
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix_normed = self._matrix / norms
        self._raw = word_vecs

    def __getitem__(self, word: str) -> np.ndarray:
        return self._raw[word]

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Return the *topn* words with highest cosine similarity to *word*."""
        if word not in self._raw:
            return []
        v = self._raw[word]
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return []
        # Batch cosine similarity: dot product against normalised matrix
        sims = self._matrix_normed @ (v / norm_v)
        idx = np.argsort(sims)[::-1]
        results: List[Tuple[str, float]] = []
        for i in idx:
            w = self._words[i]
            if w != word:  # exclude the query word itself
                results.append((w, float(sims[i])))
            if len(results) >= topn:
                break
        return results


class _SklearnWord2VecModel:
    """Thin wrapper that makes _SimpleWordVectors look like a gensim Word2Vec model.

    The real gensim model exposes its vectors as model.wv (a KeyedVectors object).
    This class replicates that structure so all downstream code works unchanged.
    """

    def __init__(self, wv: _SimpleWordVectors) -> None:
        self.wv = wv


def _train_sklearn_word_vectors(
    tokenized_sentences: List[List[str]],
    vector_size: int = 64,
) -> Optional[_SklearnWord2VecModel]:
    """Build word vectors using TF-IDF + TruncatedSVD (Latent Semantic Analysis).

    This is the fallback path when gensim is unavailable.

    How it works:
      1. Treat each tokenised sentence as a document and build a TF-IDF matrix
         X of shape (n_sentences, n_vocab).
      2. Transpose X to get a word × sentence matrix X.T.
      3. Apply TruncatedSVD (dimensionality reduction) to X.T so each word
         gets a *vector_size*-dimensional dense representation.
      4. L2-normalise every word vector.

    This is mathematically equivalent to word-level LSA and captures
    co-occurrence patterns similarly to Word2Vec in practice.
    """
    from sklearn.decomposition import TruncatedSVD

    docs = [" ".join(t) for t in tokenized_sentences]
    try:
        vectorizer = TfidfVectorizer(min_df=3, max_df=0.95, max_features=6000)
        X = vectorizer.fit_transform(docs)
    except ValueError:
        return None

    vocab = list(vectorizer.get_feature_names_out())
    if len(vocab) < 10:
        return None

    # SVD rank cannot exceed min(n_words, n_docs) - 1
    n_components = min(vector_size, len(vocab) - 1, X.shape[0] - 1)
    if n_components < 2:
        return None

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    word_matrix = svd.fit_transform(X.T)  # shape: (n_vocab, n_components)

    # L2-normalise each word vector so cosine similarity = dot product
    norms = np.linalg.norm(word_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    word_matrix = word_matrix / norms

    word_vecs = {w: word_matrix[i] for i, w in enumerate(vocab)}
    return _SklearnWord2VecModel(_SimpleWordVectors(word_vecs))


def train_word2vec_model(
    df: pd.DataFrame,
    max_rows: int = 4000,
    text_column: str = TEXT_COLUMN,
) -> Optional[Any]:
    """Train a word embedding model on the review corpus.

    Strategy:
      1. Sample up to *max_rows* reviews and tokenise them.
      2. Try gensim Word2Vec (skip-gram, sg=1) – the proper neural approach.
      3. If gensim is unavailable, fall back to TF-IDF + SVD (LSA).
      4. Returns None if the corpus is too small to produce meaningful vectors.

    The returned model object always exposes .wv.key_to_index,
    .wv[word], and .wv.most_similar(word), regardless of backend.
    """
    corpus = df[text_column].dropna().astype(str)
    if len(corpus) == 0:
        return None
    subset = corpus.sample(min(len(corpus), max_rows), random_state=42)
    tokenized = [tokenize_text(t) for t in subset]
    # Keep only sentences with at least 4 tokens (too-short sentences add noise)
    tokenized = [t for t in tokenized if len(t) >= 4]
    if len(tokenized) < 120:
        return None

    try:
        from gensim.models import Word2Vec as GensimW2V
        # sg=1 → skip-gram (better for infrequent words than CBOW)
        model = GensimW2V(
            sentences=tokenized,
            vector_size=64,
            window=6,       # context window of ±6 words
            min_count=3,    # ignore words appearing fewer than 3 times
            workers=1,
            seed=42,
            epochs=25,
            sg=1,
        )
        return model
    except Exception:
        pass

    # gensim unavailable – use the LSA-based fallback
    return _train_sklearn_word_vectors(tokenized, vector_size=64)


def embed_projection_data(model) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
    """Project word vectors to 2-D using t-SNE for the scatter plot visualisation.

    We take up to 250 words from the vocabulary, reduce their high-dimensional
    vectors to 2-D with t-SNE, and return the coordinates plus the word list.
    Perplexity is capped at 30 and floored at 5 to handle small vocabularies.
    """
    words = list(model.wv.key_to_index.keys())
    if len(words) < 8:
        return None, None, None
    vecs = np.array([model.wv[w] for w in words])
    top_n = min(len(words), 250)
    words = words[:top_n]
    vecs = vecs[:top_n]
    perplexity = min(30, max(5, len(words) - 1))
    projected = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
        max_iter=400,
    ).fit_transform(vecs)
    return projected, words, vecs


def nearest_words(model, word: str, top_n: int = 10):
    """Return the *top_n* nearest words to *word* in the embedding space."""
    if model is None:
        return []
    try:
        return model.wv.most_similar(word, topn=top_n)
    except Exception:
        return []


def cosine_and_euclidean_distance(model, w1: str, w2: str) -> Optional[Tuple[float, float]]:
    """Compute both cosine similarity and Euclidean distance between two word vectors.

    Used in the 'Distance check' display under the nearest-words table.
    Returns None if either word is not in the model vocabulary.
    """
    if model is None:
        return None
    if w1 not in model.wv.key_to_index or w2 not in model.wv.key_to_index:
        return None
    v1 = model.wv[w1]
    v2 = model.wv[w2]
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    euc = float(np.linalg.norm(v1 - v2))
    return cos, euc


# ── Classical supervised text classifier ─────────────────────────────────

def train_text_classifier(
    df: pd.DataFrame,
    target: str,
    algo: str = "Logistic Regression",
    max_rows: int = 6000,
    text_column: str = TEXT_COLUMN,
):
    """Train a TF-IDF + classifier pipeline for a single prediction target.

    Supported targets:
      'note'      → 5-class star-rating prediction  (1–5)
      'sentiment' → 3-class sentiment classification (positive/neutral/negative)
      'subject'   → 6-class subject category prediction

    Supported algorithms:
      'Logistic Regression' – linear model; provides .coef_ for explanation
      'Naive Bayes'         – MultinomialNB; fast baseline, works well on counts

    Pipeline:
      TfidfVectorizer(unigrams + bigrams, max 25k features)
        → Logistic Regression / Naive Bayes

    Returns a result dict with: model, accuracy, classification report,
    confusion matrix, error examples, and raw predictions for further analysis.
    Returns None if the dataset is too small to train reliably.
    """
    data = df[[text_column, target]].dropna().copy()
    data = data[data[text_column].str.len() > 2].copy()
    if len(data) < 150:
        return None

    # Ensure labels are strings for consistent handling across all three tasks
    if target == "note":
        data[target] = data[target].astype(int).astype(str)
    else:
        data[target] = data[target].astype(str)
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)

    X = data[text_column].astype(str)
    y = data[target]

    # Use stratified split only when every class has at least 2 samples
    stratify = None
    counts = y.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,    # 80 / 20 split
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )

    if algo == "Naive Bayes":
        classifier = MultinomialNB()
    else:
        # class_weight='balanced' compensates for the unequal class distribution
        # (there are many more 1-star and 5-star reviews than 3-star reviews)
        classifier = LogisticRegression(
            max_iter=1200,
            n_jobs=-1,
            class_weight="balanced",
        )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=list(STOP_WORDS),
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=25000,
                ),
            ),
            ("clf", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    report = pd.DataFrame(
        classification_report(y_test, pred, output_dict=True, zero_division=0)
    ).transpose()
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    y_test_reset = y_test.reset_index(drop=True)
    pred_reset = pd.Series(pred).reset_index(drop=True)
    x_test_reset = X_test.reset_index(drop=True)
    error_mask = y_test_reset != pred_reset
    if error_mask.any():
        error_examples = pd.DataFrame(
            {
                "review": x_test_reset[error_mask].astype(str),
                "true_label": y_test_reset[error_mask],
                "predicted_label": pred_reset[error_mask],
            }
        ).reset_index(drop=True)
    else:
        error_examples = pd.DataFrame(columns=["review", "true_label", "predicted_label"])
    result = {
        "model": pipeline,
        "target": target,
        "algorithm": algo,
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "y_test": y_test_reset,
        "pred": pred_reset,
        "x_test": x_test_reset,
        "error_examples": error_examples,
        "error_rate": float(error_mask.mean()),
    }
    return result


# ── HuggingFace Transformers: zero-shot classification and QA ─────────────

@st.cache_resource(show_spinner="Loading zero-shot classifier...")
def load_zero_shot_classifier():
    """Load the facebook/bart-large-mnli zero-shot classification pipeline.

    Zero-shot classification uses Natural Language Inference (NLI): the model
    treats each candidate label as a hypothesis and scores how well the input
    text 'entails' that hypothesis.  No fine-tuning on insurance data is needed.

    Cached with @st.cache_resource so the large model is downloaded and loaded
    only once per session.  Returns None if transformers is not installed.
    """
    try:
        from transformers import pipeline

        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception:
        return None


def predict_subject_zero_shot(text: str, candidate_subjects: Optional[List[str]] = None):
    """Classify a review into one of the insurance subject categories using zero-shot NLI.

    Parameters
    ----------
    text               : raw review text (French or English)
    candidate_subjects : list of category labels to score against

    Returns (predicted_label, confidence_score).
    Falls back to ('Other', 0.0) when the model is unavailable.
    """
    if not text.strip():
        return ("Other", 0.0)
    if candidate_subjects is None:
        candidate_subjects = list(SUBJECT_KEYWORDS.keys()) + ["Other"]
    model = load_zero_shot_classifier()
    if model is None:
        return ("Other", 0.0)
    try:
        # Truncate to 2000 chars to stay within the model's token limit
        result = model(text[:2000], candidate_labels=candidate_subjects, multi_label=False)
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        if labels and scores:
            return str(labels[0]), float(scores[0])
    except Exception:
        return ("Other", 0.0)
    return ("Other", 0.0)


@st.cache_resource(show_spinner="Loading QA model...")
def load_qa_pipeline():
    """Load the DistilBERT extractive question-answering pipeline.

    The model (distilbert-base-uncased-distilled-squad) is fine-tuned on SQuAD
    and extracts a short text span from a provided context as the answer.
    Used in the retrieval-based QA section after semantic search retrieves
    the most relevant reviews.

    Returns None when transformers is not installed.
    """
    try:
        from transformers import pipeline

        return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception:
        return None


def run_qa_pipeline(question: str, docs: pd.DataFrame) -> str:
    """Answer a natural-language question using retrieved review snippets as context.

    Steps:
      1. Concatenate the French text of the top-8 retrieved reviews as context.
      2. Pass (question, context) to the DistilBERT extractive QA model.
      3. Return the extracted answer span with its confidence score.

    The context is capped at 3,500 characters to respect model token limits.
    Returns an empty string when the model is unavailable or the context is empty.
    """
    if docs.empty:
        return ""
    qa_model = load_qa_pipeline()
    if qa_model is None:
        return ""
    # Build context from the retrieved reviews' French text
    context = " ".join(
        [
            str(row.get("raw_review_fr", ""))
            for _, row in docs.head(8).iterrows()
            if str(row.get("raw_review_fr", "")).strip()
        ]
    )
    if not context.strip():
        return ""
    try:
        answer = qa_model({"question": question, "context": context[:3500]})
        if isinstance(answer, dict) and answer.get("score") is not None:
            return f"{answer.get('answer', '').strip()} (confidence={answer.get('score', 0):.3f})"
    except Exception:
        return ""
    return ""


# ── Deep learning with TensorFlow/Keras ──────────────────────────────────

def build_pretrained_embedding_matrix(tokenizer, max_words: int, requested_dim: int = 50):
    """Build an embedding weight matrix from a pre-trained gensim word vector model.

    Tries to download (in order of size/speed):
      1. glove-twitter-25        (25-dim, small, fast)
      2. glove-twitter-50        (50-dim)
      3. word2vec-google-news-300 (300-dim, large)

    For each word in the Keras tokenizer's vocabulary, looks up the pre-trained
    vector and fills the corresponding row in the weight matrix.
    Words not found in the pre-trained model keep their zero initialisation.

    This matrix is then passed as the 'weights' argument to the Keras Embedding
    layer with trainable=False so the pre-trained vectors are frozen during training.

    Returns (matrix, model_name) or (None, None) if gensim is unavailable.
    """
    try:
        from gensim import downloader as api
    except Exception:
        return None, None

    model = None
    model_name = None
    candidates = [
        "glove-twitter-25",
        "glove-twitter-50",
        "word2vec-google-news-300",
    ]
    for candidate in candidates:
        try:
            model = api.load(candidate)
            model_name = candidate
            break
        except Exception:
            continue

    if model is None:
        return None, None

    embed_dim = int(getattr(model, "vector_size", requested_dim))
    # Initialise all rows to zero; only known words will be filled
    matrix = np.zeros((min(len(tokenizer.word_index) + 1, max_words), embed_dim), dtype=np.float32)
    for word, idx in tokenizer.word_index.items():
        if idx >= max_words:
            continue
        if word in model:
            matrix[idx] = np.array(model[word])
    return matrix, model_name


def train_deep_text_classifier(
    df: pd.DataFrame,
    target: str,
    max_rows: int = 5000,
    embedding_dim: int = 64,
    use_pretrained_embedding: bool = False,
    max_epochs: int = 6,
    text_column: str = TEXT_COLUMN,
):
    """Train a Keras embedding-layer text classifier.

    Architecture:
      Embedding (vocab_size × embedding_dim)
        → SpatialDropout1D(0.2)        – drops entire embedding dimensions
        → GlobalAveragePooling1D()     – reduces sequence to a single vector
        → Dense(64, relu)
        → Dropout(0.25)
        → Dense(n_classes, softmax/sigmoid)

    This is the 'bag of embeddings' approach: the GlobalAveragePooling averages
    all token embeddings to produce a fixed-size document representation.
    Simple but effective and interpretable.

    Training:
      - Adam optimiser, early stopping on val_accuracy (patience=3)
      - 80/20 stratified train/test split, 20% of training set as validation

    Pre-trained embeddings (optional):
      When *use_pretrained_embedding* is True, the function calls
      build_pretrained_embedding_matrix() to initialise the embedding layer
      with GloVe or Word2Vec weights.  The layer is then frozen (trainable=False)
      so training only updates the dense layers.

    Returns None if TensorFlow is not installed or data requirements are unmet.
    """
    data = df[[text_column, target]].dropna().copy()
    data = data[data[text_column].str.len() > 2].copy()
    if len(data) < 200:
        return None

    if target == "note":
        data[target] = data[target].astype(int).astype(str)
    else:
        data[target] = data[target].astype(str)

    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    X = data[text_column].astype(str).reset_index(drop=True)
    y = data[target].astype(str).reset_index(drop=True)

    # Encode string labels to integers for Keras
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    if len(classes) < 2:
        return None

    label_counts = pd.Series(y_encoded).value_counts()
    stratify = y_encoded if label_counts.min() >= 2 else None

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )

    X_train_text = X.iloc[train_idx]
    X_test_text = X.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    try:
        import tensorflow as tf
    except Exception:
        return None

    tf.random.set_seed(42)

    # Tokenise: map each word to an integer index, keep top 20k words
    max_words = 20000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train_text.tolist())

    # Convert text to padded integer sequences
    x_train_seq = tokenizer.texts_to_sequences(X_train_text.tolist())
    x_test_seq = tokenizer.texts_to_sequences(X_test_text.tolist())
    lengths = [len(s) for s in x_train_seq + x_test_seq]
    if not lengths:
        return None
    # Use 95th percentile length to avoid padding too many short reviews
    max_len = int(np.percentile(lengths, 95))
    max_len = max(12, min(max_len, 220))

    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train_seq,
        maxlen=max_len,
        padding="post",    # pad/truncate at the end of the sequence
        truncating="post",
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test_seq,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    pretrained_matrix = None
    if use_pretrained_embedding:
        pretrained_matrix, _ = build_pretrained_embedding_matrix(
            tokenizer=tokenizer,
            max_words=vocab_size,
            requested_dim=embedding_dim,
        )
        if pretrained_matrix is not None:
            # Adjust embedding_dim to match the pre-trained vector size
            embedding_dim = pretrained_matrix.shape[1]

    # Build the embedding layer (trainable or frozen pre-trained)
    if pretrained_matrix is None:
        embed_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            name="embedding",
        )
    else:
        embed_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            weights=[pretrained_matrix],
            trainable=False,   # freeze pre-trained weights
            name="embedding",
        )

    # Choose output layer and loss based on number of classes
    num_classes = len(classes)
    if num_classes == 2:
        out_units = 1
        out_act = "sigmoid"
        out_loss = "binary_crossentropy"
        y_train_ready = y_train.astype(np.float32)
        y_test_ready = y_test.astype(np.float32)
    else:
        out_units = num_classes
        out_act = "softmax"
        out_loss = "sparse_categorical_crossentropy"
        y_train_ready = y_train.astype(np.int32)
        y_test_ready = y_test.astype(np.int32)

    # Assemble the sequential model
    deep_model = tf.keras.Sequential(
        [
            embed_layer,
            tf.keras.layers.SpatialDropout1D(0.2),      # regularisation on embedding space
            tf.keras.layers.GlobalAveragePooling1D(),    # mean of all token embeddings
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(out_units, activation=out_act),
        ]
    )

    deep_model.compile(
        optimizer="adam",
        loss=out_loss,
        metrics=["accuracy"],
    )

    # Early stopping to avoid overfitting on small datasets
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = deep_model.fit(
        x_train,
        y_train_ready,
        validation_split=0.2,
        epochs=max_epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    probs = deep_model.predict(x_test, verbose=0)
    if num_classes == 2:
        pred = (probs.ravel() >= 0.5).astype(int)
        pred = pred.astype(int)
    else:
        pred = np.argmax(probs, axis=1)

    y_test_str = le.inverse_transform(y_test_ready.astype(int))
    pred_str = le.inverse_transform(pred.astype(int))
    acc = accuracy_score(y_test_str, pred_str)
    report = pd.DataFrame(
        classification_report(
            y_test_str,
            pred_str,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    cm = confusion_matrix(y_test_str, pred_str, labels=classes)

    errors = y_test_str != pred_str
    if np.any(errors):
        error_examples = pd.DataFrame(
            {
                "review": X_test_text.reset_index(drop=True)[errors],
                "true_label": y_test_str[errors],
                "predicted_label": pred_str[errors],
            }
        )
    else:
        error_examples = pd.DataFrame(columns=["review", "true_label", "predicted_label"])

    result = {
        "model": deep_model,
        "target": target,
        "tokenizer": tokenizer,
        "label_encoder": le,
        "classes": classes,
        "accuracy": float(acc),
        "labels": classes,
        "history": pd.DataFrame(history.history),
        "confusion_matrix": cm,
        "report": report,
        "error_examples": error_examples.reset_index(drop=True),
        "error_rate": float(np.mean(errors)),
        "pretrained": use_pretrained_embedding,
        "max_len": max_len,
        "embedding_dim": embedding_dim,
        "used_epochs": len(history.history.get("loss", [])),
        "y_test": y_test_str,
        "pred": pred_str,
        "x_test": X_test_text.reset_index(drop=True),
    }
    return result


def export_embedding_tensorboard_bundle(model, tokenizer, max_words: int = 1000):
    try:
        import tensorflow as tf
    except Exception:
        return None

    embedding_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding)]
    if not embedding_layers:
        return None

    layer = embedding_layers[0]
    weights = layer.get_weights()
    if not weights:
        return None
    matrix = weights[0]
    if matrix is None:
        return None

    word_index = tokenizer.word_index
    id2word = {idx: w for w, idx in word_index.items()}
    selected_words = [id2word[i] for i in range(1, min(len(id2word), max_words) + 1) if id2word.get(i)]
    vectors = matrix[1 : len(selected_words) + 1]
    if len(vectors) != len(selected_words):
        selected_words = selected_words[: len(vectors)]

    run_dir = TB_LOG_DIR / f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = run_dir / "vectors.tsv"
    metadata_path = run_dir / "metadata.tsv"
    np.savetxt(vectors_path, vectors, delimiter="\t")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for token in selected_words:
            f.write(f"{token}\n")

    projector = run_dir / "projector_config.pbtxt"
    with open(projector, "w", encoding="utf-8") as f:
        f.write("embeddings {\n")
        f.write("  tensor_name: \"embedding\"\n")
        f.write("  metadata_path: \"metadata.tsv\"\n")
        f.write("}\n")

    return run_dir


def plot_training_history(history_df: pd.DataFrame):
    if history_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    if "loss" in history_df:
        ax.plot(history_df["loss"], label="train_loss")
    if "val_loss" in history_df:
        ax.plot(history_df["val_loss"], label="val_loss")
    ax.set_title("Embedding model training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def build_model_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, bundle in results.items():
        if not bundle:
            continue
        report = bundle.get("report")
        if report is None or report.empty:
            continue
        def _get(row: str, col: str) -> Optional[float]:
            if row in report.index and col in report.columns:
                try:
                    return float(report.loc[row, col])
                except Exception:
                    return None
            return None

        rows.append(
            {
                "model": model_name,
                "accuracy": float(bundle.get("accuracy", 0.0)),
                "precision_macro": _get("macro avg", "precision"),
                "recall_macro": _get("macro avg", "recall"),
                "f1_macro": _get("macro avg", "f1-score"),
                "precision_weighted": _get("weighted avg", "precision"),
                "recall_weighted": _get("weighted avg", "recall"),
                "f1_weighted": _get("weighted avg", "f1-score"),
                "error_rate": float(bundle.get("error_rate", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_deep_embedding_space(model, tokenizer, top_words: int = 600):
    try:
        import tensorflow as tf
    except Exception:
        return None

    embedding_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding)]
    if not embedding_layers:
        return None
    layer = embedding_layers[0]
    weights = layer.get_weights()
    if not weights:
        return None
    matrix = np.array(weights[0])
    if matrix.ndim != 2:
        return None
    word_index = tokenizer.word_index
    id2word = {idx: w for w, idx in word_index.items()}
    max_words = min(top_words, matrix.shape[0] - 1)
    if max_words < 20:
        return None
    selected_idx = list(range(1, max_words + 1))
    tokens = [id2word.get(i, f"tok{i}") for i in selected_idx]
    vectors = matrix[selected_idx]
    perplexity = min(30, max(5, len(vectors) - 1))
    projected = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
        max_iter=400,
    ).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(projected[:, 0], projected[:, 1], alpha=0.35, s=12)
    for i, token in enumerate(tokens[:200]):
        ax.annotate(token, (projected[i, 0], projected[i, 1]), fontsize=7)
    ax.set_title("Deep embedding vectors projection (TSNE)")
    return fig


def explain_prediction(model_bundle: Dict[str, Any], text: str, top_n: int = 10):
    """Explain a logistic regression prediction by listing the most influential terms.

    Method – Local linear explanation:
      For the predicted class c, the logistic model assigns a coefficient w_i
      to each TF-IDF feature i.  The contribution of feature i to the decision
      is tf_idf(i) × w_i (a signed value).
      - Positive contribution  → word pushes the model towards class c
      - Negative contribution  → word pushes the model away from class c

    This is a simplified version of LIME's local explanation principle applied
    directly to the linear model's weight vector.

    Only works with Logistic Regression (which has .coef_).
    Returns (positive_terms, negative_terms) each as a list of (term, score) tuples.
    """
    if model_bundle is None or model_bundle["model"] is None:
        return [], []
    pipeline = model_bundle["model"]
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return [], []
    pred_label = pipeline.predict([text])[0]
    classes = clf.classes_
    if len(classes) == 2:
        coef = clf.coef_[0]
        # In binary logistic regression, coef_ corresponds to the positive class;
        # flip the sign when the prediction is the negative class
        if pred_label == classes[0]:
            coef = -coef
    else:
        if pred_label in classes:
            coef = clf.coef_[list(classes).index(pred_label)]
        else:
            coef = clf.coef_[0]

    x = vectorizer.transform([text])
    nz = x.nonzero()[1]   # indices of non-zero (present) features in the input
    names = vectorizer.get_feature_names_out()
    feature_scores = []
    for i in nz:
        # Contribution = TF-IDF weight × model coefficient
        score = float(x[0, i] * coef[i])
        feature_scores.append((names[i], score))
    pos = sorted([x for x in feature_scores if x[1] > 0], key=lambda e: e[1], reverse=True)[:top_n]
    neg = sorted([x for x in feature_scores if x[1] < 0], key=lambda e: e[1])[:top_n]
    return pos, neg


def prepare_inference_text(
    text: Any,
    use_advanced: bool = False,
    strip_accents_from_text: bool = False,
    add_bigrams: bool = False,
    min_token_length: int = 2,
) -> str:
    if use_advanced:
        return clean_text_advanced(
            text,
            strip_accents_from_text=strip_accents_from_text,
            add_bigrams=add_bigrams,
            min_token_length=min_token_length,
        )
    return clean_text(text)


def prediction_block(
    model_bundle: Dict[str, Any],
    text: str,
    use_advanced: bool = False,
    strip_accents_from_text: bool = False,
    add_bigrams: bool = False,
    min_token_length: int = 2,
):
    model = model_bundle["model"]
    clean = prepare_inference_text(
        text,
        use_advanced=use_advanced,
        strip_accents_from_text=strip_accents_from_text,
        add_bigrams=add_bigrams,
        min_token_length=min_token_length,
    )
    if model is None or not clean:
        return None, None, None, None, None

    pred = model.predict([clean])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean])[0]
        probs = list(zip(model.classes_, probabilities))
        probs = sorted(probs, key=lambda p: p[1], reverse=True)

    pos, neg = explain_prediction(model_bundle, clean, top_n=8)
    return pred, probs, pos, neg, clean


def semantic_search(
    df: pd.DataFrame,
    vectorizer,
    matrix,
    query: str,
    top_k: int = 8,
    insurer_filter: Optional[List[str]] = None,
    rating_range: Tuple[int, int] = (1, 5),
) -> pd.DataFrame:
    """Retrieve the most relevant reviews for a free-text query using TF-IDF cosine similarity.

    This is a standard TF-IDF Information Retrieval approach:
      1. The query is normalised with the same pipeline as the corpus.
      2. The query TF-IDF vector is computed using the pre-fitted vectorizer.
      3. Cosine similarity between the query vector and every document vector
         is computed in one matrix–vector operation.
      4. Results are filtered by optional insurer and rating constraints,
         then ranked by similarity score.

    The retrieved documents are also used as context for the QA pipeline below.
    """
    if not query.strip():
        return pd.DataFrame()
    q = normalize_text(query)
    q_vec = vectorizer.transform([q])
    # Compute cosine similarity of the query against all document vectors at once
    scores = cosine_similarity(q_vec, matrix).ravel()
    indexed = df.copy()
    indexed["search_score"] = scores

    min_rating, max_rating = rating_range
    indexed = indexed[(indexed["note"] >= min_rating) & (indexed["note"] <= max_rating)]
    if insurer_filter:
        indexed = indexed[indexed["assureur"].isin(insurer_filter)]
    indexed = indexed.sort_values("search_score", ascending=False).head(top_k * 2)
    return indexed.head(top_k)


def build_simple_summary(corpus: List[str], max_sentences: int = 4) -> str:
    """Generate an extractive summary for a list of review texts.

    Algorithm – frequency-based sentence scoring (TextRank-inspired):
      1. Concatenate all reviews into one text and split into sentences.
      2. Build a corpus-level word frequency table from cleaned tokens.
      3. Score each sentence by the average frequency of its unique words.
      4. Return the top *max_sentences* highest-scoring sentences.

    This lightweight approach requires no external models and works entirely
    with the built-in tokenizer and Counter.
    """
    corpus_text = " ".join([str(x) for x in corpus if isinstance(x, str)])
    if not corpus_text:
        return "No text available for summary."
    if len(corpus_text) < 250:
        return corpus_text
    sentences = re.split(r"(?<=[.!?])\s+|\n+", corpus_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    tokenized = [tokenize_text(s) for s in sentences]
    flat = [w for s in tokenized for w in s]
    freq = Counter(flat)   # corpus-wide word frequency
    if not freq:
        return " ".join(sentences[:max_sentences])
    scores = []
    for i, words in enumerate(tokenized):
        if not words:
            continue
        unique = set(words)
        # Score = sum of word frequencies / number of unique words in sentence
        score = sum(freq[w] for w in unique) / max(1, len(unique))
        scores.append((sentences[i], score))
    ranked = sorted(scores, key=lambda p: p[1], reverse=True)
    chosen = [x[0] for x in ranked[:max_sentences]]
    return " ".join(chosen)


def answer_question_fallback(question: str, docs: pd.DataFrame, top_n: int = 3) -> str:
    """Keyword-overlap fallback QA when the Transformer QA model is unavailable.

    Method – Jaccard-based snippet retrieval:
      1. Tokenise the question into a set of content words.
      2. For each retrieved review, split into sentences.
      3. Score each sentence by its Jaccard overlap with the question terms.
      4. Return the top *top_n* most overlapping sentences as the 'answer'.

    This is a simple lexical matching approach, not neural QA, but it
    provides a meaningful response even without transformers installed.
    """
    if docs.empty:
        return "No relevant review context found."
    q_terms = set(tokenize_text(question))
    if not q_terms:
        return "Question has no usable terms."

    snippets = []
    for idx, row in docs.iterrows():
        # Combine French and English text for best coverage
        candidate = f"{row.get('raw_review_fr', '')} {row.get('raw_review_en', '')}"
        sentences = re.split(r"(?<=[.!?])\s+|\n+", candidate)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
        for s in sentences:
            s_terms = set(tokenize_text(s))
            if not s_terms:
                continue
            # Jaccard similarity = |intersection| / |union|
            overlap = len(q_terms & s_terms) / len(q_terms | s_terms)
            snippets.append((overlap, s, row))

    if not snippets:
        return "No reliable evidence found in selected context."
    snippets = sorted(snippets, key=lambda x: x[0], reverse=True)[:top_n]
    picked = [f"{text}" for _, text, _ in snippets]
    return " ".join(picked)


def show_confusion_matrix(cm: np.ndarray, labels: List[str], title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=8)
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APPLICATION UI
# ═══════════════════════════════════════════════════════════════════════════

st.title("🧠 Insurance Reviews NLP + Streamlit App")
st.markdown(
    """
This app implements the project deliverables from the provided PDF:
data exploration, text cleaning, cleaning file export, topic modeling, embeddings and semantic similarity,
classical and deep learning supervised classification, prediction + explanation, insurer analysis,
insurer performance summaries, QA retrieval, and optional TensorBoard exports.
"""
)

# ── Load data (cached) ─────────────────────────────────────────────────────
reviews_df = load_dataset()
if reviews_df.empty:
    st.error(
        f"No data was found at {DATA_DIR}. Please place your .xlsx files in the data/ folder."
    )
    st.stop()

# ── Session state initialisation ──────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction; session_state
# persists values across re-runs within the same browser session.
if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}      # stores trained sklearn pipelines by task
if "deep_models" not in st.session_state:
    st.session_state["deep_models"] = {}         # stores trained Keras models by task
if "zero_shot_examples" not in st.session_state:
    st.session_state["zero_shot_examples"] = pd.DataFrame(
        columns=["text", "subject", "confidence", "status"]
    )
if "topic_result" not in st.session_state:
    st.session_state["topic_result"] = None      # cached NMF topic result

# ── Sidebar: global preprocessing controls ────────────────────────────────
# These settings affect ALL tabs: the same processed_df is shared everywhere.
st.sidebar.header("Project controls")
st.sidebar.subheader("Preprocessing mode")
preprocess_mode = st.sidebar.selectbox(
    "Text preprocessing",
    ["Basic cleaned", "Advanced with bi-grams"],
    index=0,
)
preprocess_strip_accents = st.sidebar.checkbox("Strip accents", value=False)
preprocess_add_bigrams = st.sidebar.checkbox("Add explicit bi-gram tokens", value=False)
preprocess_min_len = st.sidebar.slider("Minimum token length", 1, 8, 2)
use_advanced_preprocessing = preprocess_mode == "Advanced with bi-grams"

st.sidebar.caption(
    "Advanced mode applies to topic modeling, n-gram stats, Word2Vec, and classical/deep models."
)

# Apply the selected preprocessing to produce the working dataset
processed_df = build_preprocessed_dataset(
    reviews_df,
    use_advanced=use_advanced_preprocessing,
    strip_accents_from_text=preprocess_strip_accents,
    add_bigrams=preprocess_add_bigrams,
    min_token_length=preprocess_min_len,
)
if len(processed_df) != len(reviews_df):
    st.sidebar.warning("Some empty lines were removed after advanced preprocessing.")

# Training and search parameters shared across tabs
train_rows = st.sidebar.slider(
    "Rows used for model training",
    min_value=1,
    max_value=max(1, min(14000, len(processed_df))),
    value=max(1, min(6000, max(1, len(processed_df)))),
    step=10 if len(processed_df) < 500 else 100,
)
top_k_search = st.sidebar.slider(
    "Default search top-k",
    min_value=3,
    max_value=20,
    value=8,
    step=1,
)

# ── Main tab layout ────────────────────────────────────────────────────────
tab_overview, tab_explore, tab_models, tab_predict, tab_insurer, tab_deep = st.tabs(
    [
        "Overview",
        "Exploration & NLP tasks",
        "Supervised Models",
        "Prediction + explanation",
        "Insurer analysis + Retrieval",
        "Deep learning + deliverables",
    ]
)


# ── TAB 1: Overview ───────────────────────────────────────────────────────
# Shows high-level corpus statistics, example cleaned rows, a spelling
# correction demo, and the download button for the deliverable dataset file.
with tab_overview:
    st.subheader("Dataset and cleaning overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reviews", f"{len(reviews_df):,}")
    c2.metric("Avg star rating", f"{reviews_df['note'].mean():.2f}")
    c3.metric("Insurers", f"{reviews_df['assureur'].nunique():,}")
    c4.metric("Products", f"{reviews_df['produit'].nunique():,}")

    st.markdown("### Key distributions")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Rating distribution")
        rating_counts = reviews_df["note"].value_counts().sort_index()
        st.bar_chart(rating_counts)
    with right:
        st.markdown("#### Reviews by insurer (top 10)")
        top_insurers = reviews_df["assureur"].value_counts().head(10)
        st.bar_chart(top_insurers)

    st.markdown("### Example cleaned rows")
    sample_cols = [
        "note",
        "assureur",
        "produit",
        "raw_review_fr",
        "text_for_model",
        "sentiment",
        "subject",
        "source_file",
    ]
    st.dataframe(
        reviews_df[sample_cols].head(8),
        width="stretch",
        hide_index=True,
    )

    if st.toggle("Run a small spelling correction check", value=False):
        st.caption(
            "Spell correction is shown for first 20 French review texts (best effort only)."
        )
        sample = reviews_df["text_for_model"].head(20).tolist()
        corrected = []
        for line in sample:
            if not line:
                corrected.append("")
                continue
            try:
                from textblob import TextBlob

                corrected.append(str(TextBlob(line).correct()))
            except Exception:
                corrected.append(line)
        st.dataframe(
            pd.DataFrame({"original": sample, "textblob_corrected": corrected}),
            hide_index=True,
        )

    st.markdown("### Deliverable-ready cleaned dataset export")
    st.caption(
        "Generate a multi-column cleaned file with optional corrections and translations for submission."
    )
    st.caption(
        "Note: when translation filling is enabled, at most 500 empty rows are translated at a time."
    )
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        apply_spell = st.checkbox("Apply spelling correction columns", value=False)
    with exp_col2:
        apply_translate = st.checkbox("Fill missing translations (best effort)", value=False)
    if st.button("Create cleaned dataset copy"):
        with st.spinner("Preparing cleaned dataset file"):
            export_df = build_delivery_dataset(
                reviews_df,
                apply_spell_check=apply_spell,
                apply_translation=apply_translate,
            )
            csv_buffer = export_df.to_csv(index=False).encode("utf-8")
            st.success(f"Dataset ready: {len(export_df)} rows.")
            st.download_button(
                "Download cleaned CSV",
                csv_buffer,
                file_name=f"insurance_reviews_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="cleaned_reviews")
            excel_bytes = excel_buffer.getvalue()
            st.download_button(
                "Download cleaned Excel",
                excel_bytes,
                file_name=f"insurance_reviews_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ── TAB 2: Exploration & NLP tasks ───────────────────────────────────────
# Unsupervised analysis: n-gram frequencies, NMF topic modelling,
# Word2Vec training and t-SNE visualisation, zero-shot classification,
# and an extractive insurer summary generator.
with tab_explore:
    st.subheader("Data exploration and unsupervised features")
    left, right = st.columns(2)
    with left:
        ngram_n = st.selectbox("N-gram size", [1, 2, 3], index=1)
        top_k = st.slider("Top terms", 10, 80, 30)
        if st.button("Generate top terms"):
            ngram_df = top_ngrams(
                processed_df[TEXT_COLUMN], ngram_range=(ngram_n, ngram_n), top_k=top_k
            )
            plot_top_terms(ngram_df, f"Top {ngram_n}-grams")
            st.dataframe(ngram_df)

    with right:
        st.markdown("### Document length spread")
        st.line_chart(reviews_df["word_count"].describe()[["mean", "std", "min", "max"]])

    st.markdown("### Subject hints from rule-based labeling")
    st.dataframe(reviews_df["subject"].value_counts().rename_axis("subject").reset_index(name="count"))

    st.markdown("### Zero-shot subject classification")
    st.caption("Optional Hugging Face zero-shot labels for a single review.")
    zero_text = st.text_area(
        "Text to classify with zero-shot labels",
        value="Le délai d'indemnisation est trop long et le service client ne m'a pas rappelé.",
        height=120,
    )
    if st.button("Run zero-shot subject prediction"):
        with st.spinner("Running zero-shot classification"):
            subject_label, confidence = predict_subject_zero_shot(
                zero_text,
                candidate_subjects=list(SUBJECT_KEYWORDS.keys()) + ["Other"],
            )
            st.success(f"{subject_label} ({confidence:.4f})")
            if st.session_state["zero_shot_examples"].empty:
                st.session_state["zero_shot_examples"] = pd.DataFrame(
                    columns=["text", "subject", "confidence", "status"]
                )
            new_row = {
                "text": zero_text,
                "subject": subject_label,
                "confidence": confidence,
                "status": "ok",
            }
            st.session_state["zero_shot_examples"] = pd.concat(
                [st.session_state["zero_shot_examples"], pd.DataFrame([new_row])],
                ignore_index=True,
            ).tail(5)
            st.dataframe(st.session_state["zero_shot_examples"])

    st.markdown("### Topic modeling")
    col1, col2, col3 = st.columns(3)
    with col1:
        topic_k = st.slider("Number of topics", 2, 12, 6)
    with col2:
        words_per_topic = st.slider("Words per topic", 5, 15, 10)
    with col3:
        topic_ngrams = st.selectbox("Topic model n-gram range", ["1-1", "1-2", "2-2", "2-3"], index=1)
    topic_ngram_range = tuple(map(int, topic_ngrams.split("-")))
    if st.button("Run topic model (NMF)"):
        with st.spinner("Building topic model"):
            st.session_state["topic_result"] = run_topic_model(
                processed_df,
                n_topics=topic_k,
                top_words=words_per_topic,
                ngram_range=topic_ngram_range,
            )
    if st.session_state["topic_result"] is not None:
        st.dataframe(st.session_state["topic_result"]["topics"])

    st.markdown("### Embeddings and nearest words")
    st.info("Bonus section: Word2Vec similarity and vector visualization.")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Train Word2Vec model"):
            with st.spinner("Training Word2Vec"):
                st.session_state["word2vec_model"] = train_word2vec_model(
                    processed_df,
                    max_rows=min(4000, len(processed_df)),
                    text_column=TEXT_COLUMN,
                )
                if st.session_state["word2vec_model"] is None:
                    st.warning("Gensim not installed or too little valid data for Word2Vec.")
                else:
                    st.success("Word2Vec training done.")
    with col4:
        sample_word = st.text_input("Word for nearest neighbors", value="assurance")
    model2vec = st.session_state.get("word2vec_model")
    if model2vec is not None:
        projected, words, _ = embed_projection_data(model2vec)
        if projected is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.4, s=10)
            for i, word in enumerate(words[:40]):
                ax.annotate(word, (projected[i, 0], projected[i, 1]), fontsize=8)
            ax.set_title("Word2Vec projection (TSNE)")
            st.pyplot(fig)

        nbrs = nearest_words(model2vec, sample_word, top_n=12)
        st.markdown(f"### Nearest words to '{sample_word}'")
        if nbrs:
            st.dataframe(
                pd.DataFrame(nbrs, columns=["word", "cosine_sim"]),
                hide_index=True,
            )
            st.caption("Distance check")
            if len(nbrs) >= 2:
                second = nbrs[0][0]
                dist = cosine_and_euclidean_distance(model2vec, sample_word, second)
                if dist:
                    st.write(f"Cosine: {dist[0]:.4f}, Euclidean: {dist[1]:.4f}")
        else:
            st.write("No vectors found for this word.")

    st.markdown("### Quick summary generation")
    sample_insurer = st.selectbox(
        "Choose insurer for auto-generated summary",
        sorted(reviews_df["assureur"].dropna().unique().tolist())[:20],
    )
    if st.button("Generate insurer summary"):
        insurer_reviews = reviews_df.loc[reviews_df["assureur"] == sample_insurer, "raw_review_fr"].tolist()
        st.write(build_simple_summary(insurer_reviews, max_sentences=6))


# ── TAB 3: Supervised Models ─────────────────────────────────────────────
# Train and compare classical text classifiers (Logistic Regression vs
# Naïve Bayes) for three prediction tasks.  Shows accuracy, classification
# report, confusion matrix, and misclassified example analysis.
with tab_models:
    st.subheader("Supervised learning")
    task_to_target = {
        "Star rating": "note",
        "Sentiment": "sentiment",
        "Subject": "subject",
    }
    c1, c2, c3 = st.columns(3)
    with c1:
        task_name = st.selectbox("Task", list(task_to_target.keys()))
    with c2:
        algo = st.selectbox("Model", ["Logistic Regression", "Naive Bayes"])
    with c3:
        compare_two = st.checkbox("Train both models", value=False)

    if st.button("Train model(s) for selected task"):
        target = task_to_target[task_name]
        with st.spinner("Training model"):
            if compare_two:
                lr_result = train_text_classifier(
                    processed_df,
                    target=target,
                    algo="Logistic Regression",
                    max_rows=train_rows,
                    text_column=TEXT_COLUMN,
                )
                nb_result = train_text_classifier(
                    processed_df,
                    target=target,
                    algo="Naive Bayes",
                    max_rows=train_rows,
                    text_column=TEXT_COLUMN,
                )
                st.session_state["trained_models"][task_name] = {
                    "Logistic Regression": lr_result,
                    "Naive Bayes": nb_result,
                }
                best = None
                if lr_result and nb_result:
                    best = "Logistic Regression" if lr_result["accuracy"] >= nb_result["accuracy"] else "Naive Bayes"
                    st.success(f"Best by accuracy: {best}")
            else:
                result = train_text_classifier(
                    processed_df,
                    target=target,
                    algo=algo,
                    max_rows=train_rows,
                    text_column=TEXT_COLUMN,
                )
                st.session_state["trained_models"][task_name] = {algo: result}

    models_for_task = st.session_state["trained_models"].get(task_name, {})
    if not models_for_task:
        st.info("No model trained yet. Use the training button.")
    else:
        for name, result in models_for_task.items():
            if result is None:
                st.warning(f"{name} could not be trained.")
                continue
            st.markdown(f"### {name}")
            st.metric("Validation accuracy", f"{result['accuracy']:.3f}")
            st.caption(f"Error rate: {result['error_rate']:.3f}")
            st.dataframe(result["report"].round(3), width="stretch")
            show_confusion_matrix(
                result["confusion_matrix"],
                [str(x) for x in result["labels"]],
                f"{name} - Confusion matrix",
            )
            errors = result.get("error_examples")
            if errors is not None and not errors.empty:
                st.markdown("#### Error analysis examples")
                st.dataframe(errors.head(6), width="stretch", hide_index=True)
        comparison = build_model_summary_table(models_for_task)
        if not comparison.empty:
            st.markdown("#### Model comparison")
            st.dataframe(comparison.round(4), width="stretch")
            st.line_chart(
                comparison.set_index("model")[
                    ["accuracy", "precision_macro", "recall_macro", "f1_macro", "error_rate"]
                ]
            )

    st.markdown("### Global supervised model leaderboard")
    global_rows = []
    for task_name, model_bundle_map in st.session_state["trained_models"].items():
        summary = build_model_summary_table(model_bundle_map).assign(task=task_name)
        if not summary.empty:
            global_rows.append(summary)
    if global_rows:
        global_board = pd.concat(global_rows, ignore_index=True)
        st.dataframe(global_board[["task"] + [c for c in global_board.columns if c != "task"]].round(4))


# ── TAB 4: Prediction + Explanation ──────────────────────────────────────
# Interactive prediction for a user-typed review.  Runs all three models
# simultaneously and explains the logistic regression decision by listing
# the words with the highest positive and negative contribution scores.
with tab_predict:
    st.subheader("Interactive prediction with explanation")
    user_text = st.text_area(
        "Enter a review text",
        value="Le service client a répondu rapidement et le prix de mon assurance est correct.",
        height=160,
    )
    use_default_trained = st.checkbox("Use default auto-trained models", value=True)
    if st.button("Run prediction"):
        if use_default_trained and not st.session_state["trained_models"]:
            with st.spinner("Training all three default models in a small configuration"):
                for t, target in task_to_target.items():
                    st.session_state["trained_models"][t] = {
                        "Logistic Regression": train_text_classifier(
                            processed_df,
                            target=target,
                            algo="Logistic Regression",
                            max_rows=min(4500, len(processed_df)),
                            text_column=TEXT_COLUMN,
                        )
                    }
        if not user_text.strip():
            st.warning("Please enter a review.")
        else:
            cleaned_input = prepare_inference_text(
                user_text,
                use_advanced=use_advanced_preprocessing,
                strip_accents_from_text=preprocess_strip_accents,
                add_bigrams=preprocess_add_bigrams,
                min_token_length=preprocess_min_len,
            )
            for task_name in task_to_target:
                if not st.session_state["trained_models"].get(task_name):
                    continue
                result_map = st.session_state["trained_models"][task_name]
                model_bundle = result_map.get("Logistic Regression") or next(iter(result_map.values()))
                if model_bundle is None:
                    continue
                prediction, probabilities, pos, neg, _ = prediction_block(
                    model_bundle,
                    user_text,
                    use_advanced=use_advanced_preprocessing,
                    strip_accents_from_text=preprocess_strip_accents,
                    add_bigrams=preprocess_add_bigrams,
                    min_token_length=preprocess_min_len,
                )
                if prediction is None:
                    continue
                st.markdown(f"#### {task_name}")
                st.success(f"Prediction: {prediction}")
                if probabilities:
                    prob_df = pd.DataFrame(probabilities, columns=["label", "probability"])
                    prob_df["probability"] = prob_df["probability"].astype(float)
                    prob_df = prob_df.sort_values("probability", ascending=False)
                    st.bar_chart(prob_df.set_index("label")["probability"])
                if pos:
                    st.caption("Positive evidence")
                    st.table(pd.DataFrame(pos, columns=["term", "contribution"]).head(10))
                if neg:
                    st.caption("Negative evidence")
                    st.table(pd.DataFrame(neg, columns=["term", "contribution"]).head(10))

            st.markdown("### Cleaned version used for modeling")
            st.code(cleaned_input)


# ── TAB 5: Insurer Analytics + Retrieval ─────────────────────────────────
# Per-insurer performance statistics, subject breakdown pivot table,
# TF-IDF semantic search over the review corpus, and a retrieval-based
# QA section (Transformer extractive QA or keyword-overlap fallback).
with tab_insurer:
    st.subheader("Insurer intelligence dashboard")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        insurer_filter = st.multiselect(
            "Filter by insurer",
            options=sorted(reviews_df["assureur"].dropna().unique().tolist()),
            default=[],
        )
    with summary_col2:
        rating_limits = st.slider("Filter by rating", 1, 5, (1, 5), step=1)

    working = processed_df.copy()
    min_rating, max_rating = rating_limits
    working = working[(working["note"] >= min_rating) & (working["note"] <= max_rating)]
    if insurer_filter:
        working = working[working["assureur"].isin(insurer_filter)]

    st.markdown("### Summary by insurer")
    insurer_stats = (
        working.groupby("assureur")
        .agg(
            total_reviews=("note", "count"),
            average_rating=("note", "mean"),
        )
        .sort_values("average_rating", ascending=False)
    )
    insurer_stats["average_rating"] = insurer_stats["average_rating"].round(2)
    st.dataframe(insurer_stats, width="stretch")

    if insurer_filter:
        st.markdown("### Subject by insurer (selected insurers)")
        subject_matrix = (
            working.pivot_table(
                index="assureur",
                columns="subject",
                values="note",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        st.dataframe(subject_matrix, width="stretch")
    st.markdown("### Average rating by insurer and subject")
    subject_avg = (
        working.pivot_table(
            index="assureur",
            columns="subject",
            values="note",
            aggfunc="mean",
        )
        .round(2)
    )
    st.dataframe(subject_avg, width="stretch")

    st.markdown("### Information retrieval + bonus QA")
    vectorizer, matrix = build_search_index(working)
    query = st.text_input(
        "Search in reviews (semantic retrieval)",
        "problème au moment d'une panne et d'indemnisation",
    )
    if st.button("Search reviews"):
        top_k = top_k_search
        results = semantic_search(
            working,
            vectorizer=vectorizer,
            matrix=matrix,
            query=query,
            top_k=top_k,
            insurer_filter=insurer_filter if insurer_filter else None,
            rating_range=rating_limits,
        )
        if results.empty:
            st.warning("No results found.")
        else:
            st.dataframe(
                results[
                    [
                        "search_score",
                        "note",
                        "assureur",
                        "produit",
                        "raw_review_fr",
                    ]
                ].head(top_k),
                width="stretch",
            )
            question = st.text_input("Ask a question over retrieved reviews", "What do customers say about delays?")
            if st.button("Run QA / RAG"):
                answer = run_qa_pipeline(question, results)
                if not answer:
                    answer = answer_question_fallback(question, results)
                    st.info(answer)
                else:
                    st.success(answer)


# ── TAB 6: Deep Learning + TensorBoard ───────────────────────────────────
# Trains a Keras sequential model with a trainable (or frozen pre-trained)
# embedding layer.  Displays training curves, confusion matrix, error
# analysis, and exports embedding vectors for TensorBoard Projector.
with tab_deep:
    st.subheader("Deep learning, pre-trained embeddings, and TensorBoard")
    deep_task_map = {
        "Star rating": "note",
        "Sentiment": "sentiment",
        "Subject": "subject",
    }
    deep_default_rows = max(1, min(5000, len(processed_df)))
    deep_min_rows = max(1, min(800, deep_default_rows))
    deep_step = 10 if deep_default_rows < 500 else 200
    col1, col2, col3 = st.columns(3)
    with col1:
        deep_task = st.selectbox("Deep task", list(deep_task_map.keys()))
    with col2:
        embedding_dim = st.slider("Embedding dim", 16, 128, 64, step=8)
    with col3:
        deep_rows = st.slider(
            "Rows used for deep model",
            min_value=deep_min_rows,
            max_value=min(12000, len(processed_df)),
            value=deep_default_rows,
            step=deep_step,
        )
    pretrain = st.toggle("Use pre-trained embeddings (optional)")
    use_epochs = st.slider("Epochs", 2, 12, 6, step=1)

    if st.button("Train deep learning model"):
        with st.spinner("Training TensorFlow/Keras text classifier"):
            deep_bundle = train_deep_text_classifier(
                processed_df,
                target=deep_task_map[deep_task],
                max_rows=deep_rows,
                embedding_dim=embedding_dim,
                use_pretrained_embedding=pretrain,
                max_epochs=use_epochs,
                text_column=TEXT_COLUMN,
            )
            if deep_bundle is None:
                st.error("Could not train deep model (TensorFlow or data requirements missing).")
            else:
                st.session_state["deep_models"][deep_task] = deep_bundle
                st.success("Deep model trained.")

    deep_bundle = st.session_state["deep_models"].get(deep_task)
    if deep_bundle:
        st.metric("Deep model accuracy", f"{deep_bundle['accuracy']:.3f}")
        st.metric("Error rate", f"{deep_bundle['error_rate']:.3f}")
        st.metric("Used embedding dim", str(deep_bundle.get("embedding_dim", "?")))
        st.caption(f"Training epochs completed: {deep_bundle.get('used_epochs', 0)}")
        st.dataframe(deep_bundle["report"].round(3), width="stretch")
        show_confusion_matrix(
            deep_bundle["confusion_matrix"],
            [str(x) for x in deep_bundle["labels"]],
            "Deep model confusion matrix",
        )
        deep_errors = deep_bundle.get("error_examples")
        if deep_errors is not None and not deep_errors.empty:
            st.markdown("### Deep model error analysis")
            st.dataframe(deep_errors.head(8), width="stretch", hide_index=True)
        if deep_bundle.get("history") is not None and not deep_bundle["history"].empty:
            fig = plot_training_history(deep_bundle["history"])
            if fig is not None:
                st.pyplot(fig)
                if st.button("Visualize deep embedding space", key="deep_embedding_plot"):
                    embed_fig = plot_deep_embedding_space(
                        deep_bundle["model"],
                        deep_bundle["tokenizer"],
                        top_words=min(600, len(deep_bundle["tokenizer"].word_index)),
                    )
                    if embed_fig is not None:
                        st.pyplot(embed_fig)
                    else:
                        st.warning("No embedding vector map available for visualization.")

        deep_summary_rows = []
        for task_key, deep_item in st.session_state["deep_models"].items():
            if not deep_item:
                continue
            metrics = {
                "task": task_key,
                "accuracy": float(deep_item.get("accuracy", 0.0)),
                "error_rate": float(deep_item.get("error_rate", 0.0)),
                "used_epochs": int(deep_item.get("used_epochs", 0)),
                "pretrained_embedding": bool(deep_item.get("pretrained", False)),
                "embedding_dim": int(deep_item.get("embedding_dim", 0)),
                "max_len": int(deep_item.get("max_len", 0)),
            }
            deep_summary_rows.append(metrics)
        if deep_summary_rows:
            st.markdown("#### Deep models leaderboard")
            st.dataframe(pd.DataFrame(deep_summary_rows), width="stretch")

        if st.button("Export embeddings for TensorBoard"):
            with st.spinner("Saving projector files"):
                tb_path = export_embedding_tensorboard_bundle(
                    deep_bundle["model"],
                    deep_bundle["tokenizer"],
                    max_words=1500,
                )
                if tb_path:
                    st.success(f"TensorBoard files created at: {tb_path}")
                    st.code(f"tensorboard --logdir \"{tb_path.parent}\"")
                else:
                    st.warning("TensorBoard export failed.")

