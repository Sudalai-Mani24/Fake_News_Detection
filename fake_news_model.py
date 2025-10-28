"""
fake_news_model.py

Provides:
- Baseline model: TF-IDF + LogisticRegression (fast)
- Optional transformer model (BERT/FinBERT) via HuggingFace (accurate, slower)
- Train / save / load / predict utilities
- Integration helper to classify a pandas.DataFrame of news articles

Dependencies (add to requirements.txt):
- scikit-learn
- pandas
- joblib
- sqlalchemy
- psycopg2-binary
- transformers (optional, for transformer model)
- torch (if using transformers and not running on CPU-only fallback)
"""

import os
import logging
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# sklearn baseline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Optional transformer
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Database (for optional saving of predictions)
from sqlalchemy import create_engine, text

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration / Defaults
# ------------------------------
DEFAULT_MODEL_DIR = "models"
BASELINE_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_logreg.joblib")
VECTORIZER_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.joblib")
TRANSFORMER_MODEL_NAME = "yiyanghkust/finbert-tone"  # optional; change if you prefer another FinBERT variant
TRANSFORMER_DIR = os.path.join(DEFAULT_MODEL_DIR, "transformer_model")

# Ensure model directory exists
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

# ------------------------------
# Baseline: TF-IDF + LogisticRegression
# ------------------------------
def train_baseline(df: pd.DataFrame,
                   text_col: str = "text",
                   label_col: str = "label",
                   test_size: float = 0.2,
                   random_state: int = 42,
                   save_model: bool = True) -> Dict[str, Any]:
    """
    Train a baseline TF-IDF + Logistic Regression classifier.
    df: dataframe with columns [text_col, label_col] (label can be 0/1 or 'FAKE'/'REAL')
    Returns: dict with pipeline, vectorizer, model, metrics
    """
    logger.info("Preparing labels...")
    # standardize labels to 0/1
    df = df.dropna(subset=[text_col, label_col]).copy()
    y_raw = df[label_col].astype(str).str.lower().map(lambda x: 1 if x in ["real", "true", "1", "genuine"] else 0)
    X = df[text_col].astype(str).values
    y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Baseline accuracy: {acc:.4f}")
    logger.info("Classification report (test):")
    logger.info(classification_report(y_test, y_pred))

    if save_model:
        logger.info(f"Saving vectorizer -> {VECTORIZER_PATH}")
        joblib.dump(vectorizer, VECTORIZER_PATH)
        logger.info(f"Saving model -> {BASELINE_MODEL_PATH}")
        joblib.dump(model, BASELINE_MODEL_PATH)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": acc,
        "report": report
    }

def load_baseline() -> Dict[str, Any]:
    """Load TF-IDF vectorizer + LogisticRegression model from disk."""
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(BASELINE_MODEL_PATH):
        raise FileNotFoundError("Baseline model or vectorizer not found. Train first.")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(BASELINE_MODEL_PATH)
    logger.info("Baseline model & vectorizer loaded.")
    return {"vectorizer": vectorizer, "model": model}

def predict_baseline(texts: Union[str, List[str]], loaded: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Predict with baseline model.
    texts: str or list of str
    loaded: optional dict from load_baseline()
    Returns list of {'label': 'REAL'/'FAKE', 'score': prob_of_real}
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    if loaded is None:
        loaded = load_baseline()

    vectorizer = loaded["vectorizer"]
    model = loaded["model"]

    X = vectorizer.transform([str(t) for t in texts])
    probs = model.predict_proba(X)  # assumes classes [0,1] where 1=real
    # sklearn's predict_proba columns correspond to model.classes_
    # find index of class 1
    classes = model.classes_
    if 1 in classes:
        idx_real = int(np.where(classes == 1)[0])
    else:
        # fallback: assume second column is positive class
        idx_real = 1

    results = []
    for i, t in enumerate(texts):
        prob_real = float(probs[i][idx_real])
        label = "REAL" if prob_real >= 0.5 else "FAKE"
        results.append({"text": t, "label": label, "prob_real": prob_real})

    return results[0] if single else results

# ------------------------------
# Transformer-based classifier (optional)
# ------------------------------
class TransformerWrapper:
    def __init__(self, model_name: str = TRANSFORMER_MODEL_NAME, local_dir: Optional[str] = TRANSFORMER_DIR, device: int = -1):
        """
        device: -1 for CPU, or GPU id (0,1,...). HuggingFace pipeline will handle.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available. Install transformers and torch to use transformer model.")
        self.model_name = model_name
        self.local_dir = local_dir
        self.device = device
        self._pipeline = None

    def load(self, force_download: bool = False):
        logger.info(f"Loading transformer model '{self.model_name}' (force_download={force_download})...")
        if not os.path.exists(self.local_dir) or force_download:
            # we will rely on .from_pretrained and cache in local dir
            os.makedirs(self.local_dir, exist_ok=True)
            # the pipeline will download the model and tokenizer
        self._pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, return_all_scores=False, device=self.device)
        logger.info("Transformer pipeline ready.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        if self._pipeline is None:
            self.load()
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        outs = self._pipeline(texts, truncation=True)
        # outputs like [{'label':'LABEL_1','score':0.98}, ...]
        results = []
        for t, o in zip(texts, outs):
            label = o.get("label")
            score = float(o.get("score", 0.0))
            # mapping label names to REAL/FAKE might be model dependent.
            # common FinBERT variants output POS/NEG or 0/1. We'll attempt a heuristic:
            mapped_label = "REAL" if label.lower().startswith("real") or label.lower().startswith("true") or label.lower().endswith("_1") or label.lower().endswith("1") else "FAKE"
            results.append({"text": t, "label": mapped_label, "score": score, "raw_label": label})
        return results[0] if single else results

# ------------------------------
# Helpers to apply model to dataframe and save predictions to DB
# ------------------------------
def classify_dataframe_with_baseline(df: pd.DataFrame, text_col: str = "title", model_loaded: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Given a DataFrame with a text column, return a copy with new columns:
    - predicted_label (REAL/FAKE)
    - prob_real (float)
    """
    if model_loaded is None:
        model_loaded = load_baseline()
    texts = df[text_col].astype(str).tolist()
    preds = predict_baseline(texts, loaded=model_loaded)
    # preds is list of dicts
    pred_labels = [p["label"] for p in preds]
    pred_probs = [p["prob_real"] for p in preds]
    out = df.copy()
    out["predicted_label"] = pred_labels
    out["prob_real"] = pred_probs
    out["predicted_at"] = datetime.utcnow()
    return out

def write_predictions_to_db(df: pd.DataFrame,
                            table_name: str = "finance_news",
                            db_uri: Optional[str] = None,
                            key_column: Optional[str] = "id"):
    """
    Write predicted_label & prob_real back to postgres.
    db_uri: SQLAlchemy URI, e.g. 'postgresql+psycopg2://user:pass@localhost:5432/finance_news_db'
    key_column: column that uniquely identifies a row in the target table (e.g. id or url)
    """
    if db_uri is None:
        raise ValueError("db_uri must be provided to write back to DB.")

    engine = create_engine(db_uri)
    conn = engine.connect()

    # we will upsert predicted columns per row matching key_column
    # This implementation issues an UPDATE for each row; for larger scale, consider bulk copy or using temp table + upsert
    for _, row in df.iterrows():
        key_val = row[key_column]
        predicted_label = row.get("predicted_label")
        prob_real = float(row.get("prob_real", 0.0))
        predicted_at = row.get("predicted_at", datetime.utcnow())
        sql = text(f"""
            UPDATE {table_name}
            SET predicted_label = :predicted_label,
                prob_real = :prob_real,
                predicted_at = :predicted_at
            WHERE {key_column} = :key_val;
        """)
        try:
            conn.execute(sql, {"predicted_label": predicted_label, "prob_real": prob_real, "predicted_at": predicted_at, "key_val": key_val})
        except Exception as e:
            logger.exception(f"Failed to update row {key_val}: {e}")

    conn.close()
    logger.info("Predictions written to database.")

# ------------------------------
# Convenience CLI / Example usage
# ------------------------------
def example_train_and_save(csv_path: str, text_col: str = "text", label_col: str = "label"):
    """
    Example helper to train baseline model from CSV and save to disk.
    CSV must contain columns: text_col, label_col
    """
    logger.info(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    res = train_baseline(df, text_col=text_col, label_col=label_col, save_model=True)
    logger.info("Training complete.")
    return res

def example_classify_csv(csv_path: str, output_csv: Optional[str] = None, text_col: str = "title", id_col: str = "id"):
    """
    Load a CSV of news (must include text_col and an id_col), classify with baseline model, and optionally save results
    """
    logger.info(f"Loading news CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df_with_preds = classify_dataframe_with_baseline(df, text_col=text_col)
    if output_csv:
        df_with_preds.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions to {output_csv}")
    return df_with_preds

# ------------------------------
# If executed as a script: quick demo CLI
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/predict fake-news model (baseline TF-IDF or transformer).")
    parser.add_argument("--mode", choices=["train", "predict", "demo_transformer"], default="predict", help="Operation mode")
    parser.add_argument("--input", help="Input CSV for train/predict (required for train/predict)")
    parser.add_argument("--text_col", default="text", help="Text column name in CSV")
    parser.add_argument("--label_col", default="label", help="Label column name (for training)")
    parser.add_argument("--output", help="Output CSV to save predictions (for predict mode)")
    parser.add_argument("--db_uri", help="If provided, write predictions back to given database URI (SQLAlchemy)")
    parser.add_argument("--table", default="finance_news", help="Table to update when db_uri provided")
    parser.add_argument("--id_col", default="id", help="Key column to match when updating DB (e.g. id)")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.input:
            raise SystemExit("Please provide --input path to labeled CSV for training.")
        example_train_and_save(args.input, text_col=args.text_col, label_col=args.label_col)
    elif args.mode == "predict":
        if not args.input:
            raise SystemExit("Please provide --input path to CSV for prediction.")
        out_df = example_classify_csv(args.input, output_csv=args.output, text_col=args.text_col, id_col=args.id_col)
        if args.db_uri:
            write_predictions_to_db(out_df, table_name=args.table, db_uri=args.db_uri, key_column=args.id_col)
        logger.info("Predict mode finished.")
    elif args.mode == "demo_transformer":
        if not TRANSFORMERS_AVAILABLE:
            raise SystemExit("Transformers not available in environment. Install transformers and torch.")
        tw = TransformerWrapper()
        tw.load()
        sample_texts = [
            "Central bank raises interest rates by 50 basis points",
            "Breaking: CEO arrested in corruption scandal — stock to plunge 60%!"
        ]
        preds = tw.predict(sample_texts)
        for p in preds:
            logger.info(p)
    else:
        logger.info("No valid mode chosen. Use --help for options.")
