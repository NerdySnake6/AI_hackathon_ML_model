"""Script to train and serialize the ML domain classifier using scikit-learn."""

import argparse
import csv
import logging
from pathlib import Path

import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from app.preprocessing.normalizer import build_normalized_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/imdb_labeled_queries.csv")
DEFAULT_MODEL_PATH = Path("outputs/domain_classifier.pkl")

def load_data(filepath: Path) -> tuple[list[str], list[bool]]:
    """Load query data and boolean labels indicating video content."""
    X = []
    y = []
    with filepath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_text = row["query_text"].strip()
            # Support both prototype and IMDB generated datasets
            video_flag = row.get("expected_is_prof_video") or row.get("is_prof_video", "false")
            is_video = video_flag.lower() == "true"
            
            # Use identical preprocessing as inference 
            normalized = build_normalized_query(query_text).normalized_text
            
            if normalized:
                X.append(normalized)
                y.append(is_video)
                
    return X, y

def train_and_save_model(data_path: Path, output_path: Path) -> None:
    """Train the TF-IDF + LogisticRegression model and save to disk."""
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        return

    logger.info("Loading dataset...")
    X, y = load_data(data_path)
    if not X:
        logger.error("Dataset is empty after normalization!")
        return
        
    logger.info(f"Loaded {len(X)} examples ({sum(y)} positive / {len(y) - sum(y)} negative).")

    # The pipeline translates raw normalized text into character n-grams and word tokens,
    # then applies an SGDClassifier optimized for fast logistical regression loss.
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
            )
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss", # Provides predict_proba capability
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    logger.info("Training pipeline...")
    pipeline.fit(X, y)
    
    # Evaluate loosely on train set since data is small
    predictions = pipeline.predict(X)
    report = classification_report(y, predictions, target_names=["Non-Video", "Video"])
    logger.info(f"Training set evaluation:\n{report}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Model serialized successfully to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Domain Classifier ML module.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to labeled data CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Destination path for .pkl model"
    )
    args = parser.parse_args()
    train_and_save_model(args.data, args.output)

if __name__ == "__main__":
    main()
