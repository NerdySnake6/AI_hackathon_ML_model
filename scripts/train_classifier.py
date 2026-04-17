"""Script to train the ML models for Mediascope hackathon."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/raw/mediascope/train.csv")
DEFAULT_TYPEQUERY_MODEL_PATH = Path("outputs/typequery_classifier.pkl")
DEFAULT_CONTENTTYPE_MODEL_PATH = Path("outputs/contenttype_classifier.pkl")
DEFAULT_TITLES_PATH = Path("outputs/titles.json")

def train_and_save_models(
    data_path: Path, 
    typequery_output: Path, 
    contenttype_output: Path, 
    titles_output: Path
) -> None:
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        return

    logger.info("Loading Mediascope dataset...")
    df_mediascope = pd.read_csv(data_path)
    df_mediascope = df_mediascope.dropna(subset=['QueryText'])
    
    logger.info("Loading IMDb synthetic dataset...")
    imdb_path = Path("data/imdb_labeled_queries.csv")
    if imdb_path.exists():
        df_imdb = pd.read_csv(imdb_path)
        
        # Map IMDb columns to Mediascope format
        df_imdb_mapped = pd.DataFrame()
        df_imdb_mapped['QueryText'] = df_imdb['query_text']
        # Handle string booleans or actual booleans
        df_imdb_mapped['TypeQuery'] = df_imdb['is_prof_video'].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)
        
        content_map = {
            'film': 'фильм',
            'series': 'сериал',
            'generic': 'прочее'
        }
        df_imdb_mapped['ContentType'] = df_imdb['content_type'].map(content_map).fillna('прочее')
        df_imdb_mapped['Title'] = None  # Synthetic queries don't have titles in the csv
        
        df = pd.concat([df_mediascope, df_imdb_mapped], ignore_index=True)
        initial_len = len(df)
        # Keep Mediascope truth if there's a collision
        df = df.drop_duplicates(subset=['QueryText'], keep='first')
        logger.info(f"Combined dataset: {len(df_mediascope)} (Mediascope) + {len(df_imdb_mapped)} (IMDb) = {initial_len} total rows")
        logger.info(f"Removed {initial_len - len(df)} duplicates. Final training rows: {len(df)}")
    else:
        df = df_mediascope
        logger.warning(f"IMDb dataset not found at {imdb_path}. Training on Mediascope data only.")

    
    # 1. Train TypeQuery Classifier
    logger.info("Training TypeQuery classifier...")
    X_tq = df['QueryText'].astype(str)
    y_tq = df['TypeQuery'].astype(int)
    
    tq_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=2,
            )
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])
    tq_pipeline.fit(X_tq, y_tq)
    logger.info(f"TypeQuery Training evaluation:\n{classification_report(y_tq, tq_pipeline.predict(X_tq))}")
    
    typequery_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tq_pipeline, typequery_output)
    
    # 2. Train ContentType Classifier
    # We only train ContentType on examples where TypeQuery = 1
    logger.info("Training ContentType classifier...")
    df_ct = df[df['TypeQuery'] == 1].copy()
    # ContentType can be missing or NaN, replace with "прочее"
    df_ct['ContentType'] = df_ct['ContentType'].fillna('прочее').astype(str)
    
    X_ct = df_ct['QueryText'].astype(str)
    y_ct = df_ct['ContentType']
    
    ct_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=2,
            )
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])
    ct_pipeline.fit(X_ct, y_ct)
    logger.info(f"ContentType Training evaluation:\n{classification_report(y_ct, ct_pipeline.predict(X_ct))}")
    
    joblib.dump(ct_pipeline, contenttype_output)
    
    # 3. Extract Titles for Fuzzy Matching
    logger.info("Extracting known titles...")
    # Get unique non-null titles
    titles = df['Title'].dropna().astype(str).unique().tolist()
    # Filter out empty strings or very short titles if necessary, but we'll keep all for now
    titles = [t.strip() for t in titles if t.strip()]
    
    with open(titles_output, 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Saved {len(titles)} unique titles to {titles_output}")
    logger.info("All models serialized successfully.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Domain Classifier ML module.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to labeled data CSV"
    )
    args = parser.parse_args()
    train_and_save_models(
        args.data, 
        DEFAULT_TYPEQUERY_MODEL_PATH, 
        DEFAULT_CONTENTTYPE_MODEL_PATH, 
        DEFAULT_TITLES_PATH
    )

if __name__ == "__main__":
    main()
