"""ContentType classifier for TypeQuery=1 queries."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing import extract_features, features_to_vector, normalize


def _prepare_positive_frame(df):
    """Return TypeQuery=1 rows with normalized labels and text."""
    pos = df[df["TypeQuery"] == 1].copy()
    pos["ContentType"] = pos["ContentType"].fillna("").astype(str)
    pos.loc[pos["ContentType"] == "", "ContentType"] = "прочее"
    pos["NormalizedQuery"] = pos["QueryText"].astype(str).map(normalize)
    return pos


def _fit_feature_bundle(texts: list[str]) -> dict[str, TfidfVectorizer]:
    """Fit sparse text feature extractors for CT classification."""
    char_tfidf = TfidfVectorizer(
        max_features=24000,
        ngram_range=(2, 5),
        analyzer="char_wb",
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    word_tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        analyzer="word",
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    char_tfidf.fit(texts)
    word_tfidf.fit(texts)
    return {
        "char_tfidf": char_tfidf,
        "word_tfidf": word_tfidf,
    }


def _transform_texts(texts: list[str], feature_bundle: dict) -> csr_matrix:
    """Transform normalized queries into a sparse hybrid feature matrix."""
    char_matrix = feature_bundle["char_tfidf"].transform(texts)
    word_matrix = feature_bundle["word_tfidf"].transform(texts)
    hand_matrix = np.array([features_to_vector(extract_features(text)) for text in texts], dtype=float)
    return hstack([char_matrix, word_matrix, csr_matrix(hand_matrix)], format="csr")


def train_ct_classifier(df):
    """Train a sparse hybrid ContentType classifier."""
    pos = _prepare_positive_frame(df)
    texts = pos["NormalizedQuery"].tolist()
    y = pos["ContentType"].astype(str).to_numpy()

    print(f"  Training on {len(pos)} positive queries")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    x_train_text, x_val_text, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.12,
        stratify=y,
        random_state=42,
    )

    feature_bundle = _fit_feature_bundle(x_train_text)
    x_train = _transform_texts(x_train_text, feature_bundle)
    x_val = _transform_texts(x_val_text, feature_bundle)

    model = LinearSVC(
        C=1.1,
        class_weight="balanced",
        dual="auto",
        max_iter=12000,
    )
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    cv_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"  Holdout macro F1: {cv_f1:.4f}")

    final_bundle = _fit_feature_bundle(texts)
    x_full = _transform_texts(texts, final_bundle)
    model.fit(x_full, y)

    return model, final_bundle


def save_artifacts(model, feature_bundle, path: str = "artifacts/ct_classifier.pkl"):
    """Persist the CT classifier and its feature bundle to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"model": model, "features": feature_bundle}, handle)


def load_artifacts(path: str = "artifacts/ct_classifier.pkl"):
    """Load the CT classifier artifact from disk."""
    if not os.path.exists(path):
        return None, None

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    if "features" in data:
        return data["model"], data["features"]

    legacy_tfidf = data.get("tfidf")
    if legacy_tfidf is None:
        return data.get("model"), None

    return data["model"], {"char_tfidf": legacy_tfidf, "word_tfidf": None}


class ContentTypeClassifier:
    """Predict ContentType for TypeQuery=1 queries."""

    VALID_LABELS = ["сериал", "фильм", "мультфильм", "мультсериал", "прочее"]

    def __init__(self, model, feature_bundle):
        """Store a trained classifier and its feature extractors."""
        self.model = model
        self.feature_bundle = feature_bundle

    def predict(self, queries: list[str]) -> list[str]:
        """Predict ContentType for a batch of raw queries."""
        return [label for label, _ in self.predict_with_margins(queries)]

    def predict_with_margins(self, queries: list[str]) -> list[tuple[str, float | None]]:
        """Predict ContentType together with a confidence-like decision margin."""
        if self.model is None or self.feature_bundle is None:
            return [("прочее", None)] * len(queries)

        normalized_queries = [normalize(query) for query in queries]

        if self.feature_bundle.get("word_tfidf") is None:
            feature_matrix = self.feature_bundle["char_tfidf"].transform(normalized_queries)
        else:
            feature_matrix = _transform_texts(normalized_queries, self.feature_bundle)

        is_dense_required = type(self.model).__name__ == "HistGradientBoostingClassifier"
        X_input = feature_matrix.toarray() if is_dense_required else feature_matrix
        preds = self.model.predict(X_input)
        normalized_preds = [pred if pred in self.VALID_LABELS else "прочее" for pred in preds]

        # Calculate margins
        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X_input)
            if getattr(decision, "ndim", 1) == 1:
                margins = [abs(float(score)) for score in decision]
            else:
                margins = []
                for row in decision:
                    ranked = np.sort(np.asarray(row, dtype=float))
                    margins.append(float(ranked[-1] - ranked[-2]) if len(ranked) >= 2 else abs(float(ranked[0])) if len(ranked) == 1 else None)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)
            margins = []
            for row in probs:
                ranked = np.sort(np.asarray(row, dtype=float))
                margins.append(float(ranked[-1] - ranked[-2]) if len(ranked) >= 2 else float(ranked[0]))
        else:
            return [(pred, None) for pred in normalized_preds]

        return list(zip(normalized_preds, margins))
