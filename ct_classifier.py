"""ContentType classifier for TypeQuery=1 queries."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing import extract_features, features_to_vector, normalize

CT_ARTIFACT_FORMAT_VERSION = 2
SUPPORTED_CT_MODEL_TYPES = {"LinearSVC"}


def _prepare_positive_frame(df):
    """Return TypeQuery=1 rows with normalized labels and text."""
    pos = df[df["TypeQuery"] == 1].copy()
    pos["ContentType"] = pos["ContentType"].fillna("").astype(str)
    pos.loc[pos["ContentType"] == "", "ContentType"] = "прочее"
    pos["NormalizedQuery"] = pos["QueryText"].astype(str).map(lambda x: normalize(x, use_lemmatization=True))
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


def augment_with_dictionary(texts: list[str], labels: list[str], franchise_dict: dict):
    """Augment training data with synthetic queries from the dictionary."""
    if not franchise_dict:
        return texts, labels

    aug_texts = list(texts)
    aug_labels = list(labels)

    for title, data in franchise_dict.items():
        ct = data.get("contentType")
        if not ct or ct not in ContentTypeClassifier.VALID_LABELS:
            continue

        # Add the normalized title itself as a query
        norm_title = normalize(title, use_lemmatization=True)
        if norm_title:
            aug_texts.append(norm_title)
            aug_labels.append(ct)

        # Deterministic augmentation keeps retraining reproducible across runs.
        noise_variants = [
            "смотреть",
            "онлайн",
            "бесплатно",
            "hd",
            "в хорошем качестве",
            "1080p",
            "",
        ]
        noise = noise_variants[sum(ord(ch) for ch in norm_title) % len(noise_variants)] if norm_title else ""

        if ct == "фильм":
            aug_texts.append(f"фильм {norm_title} {noise}".strip())
            aug_labels.append(ct)
            aug_texts.append(f"{norm_title} смотреть онлайн")
            aug_labels.append(ct)
        elif ct == "сериал":
            aug_texts.append(f"сериал {norm_title} {noise}".strip())
            aug_labels.append(ct)
            aug_texts.append(f"{norm_title} все серии {noise}".strip())
            aug_labels.append(ct)
        elif ct == "мультсериал":
            aug_texts.append(f"мультсериал {norm_title}")
            aug_labels.append(ct)
            aug_texts.append(f"мультик {norm_title} {noise}".strip())
            aug_labels.append(ct)
            aug_texts.append(f"аниме {norm_title}")
            aug_labels.append(ct)
        elif ct == "мультфильм":
            aug_texts.append(f"мультфильм {norm_title} {noise}".strip())
            aug_labels.append(ct)
            aug_texts.append(f"мультик {norm_title}")
            aug_labels.append(ct)
        elif ct == "прочее":
            aug_texts.append(f"{norm_title} смотреть")
            aug_labels.append(ct)

    return aug_texts, np.array(aug_labels)


def train_ct_classifier(df, franchise_dict: dict = None):
    """Train a sparse hybrid ContentType classifier."""
    pos = _prepare_positive_frame(df)
    texts = pos["NormalizedQuery"].tolist()
    y = pos["ContentType"].astype(str).tolist()

    if franchise_dict:
        print(f"  Augmenting CT data with {len(franchise_dict)} dictionary entries...")
        texts, y = augment_with_dictionary(texts, y, franchise_dict)
    
    y = np.array(y)
    print(f"  Training on {len(texts)} queries (original + augmented)")

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
        C=0.9,
        class_weight="balanced",
        dual="auto",
        max_iter=20000,
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
    artifact_payload = {
        "model": model,
        "features": feature_bundle,
        "artifact_format_version": CT_ARTIFACT_FORMAT_VERSION,
        "model_class": type(model).__name__ if model is not None else None,
        "sklearn_version": sklearn_version,
    }
    with open(path, "wb") as handle:
        pickle.dump(artifact_payload, handle)


def load_artifacts(path: str = "artifacts/ct_classifier.pkl"):
    """Load and validate the CT classifier artifact from disk."""
    if not os.path.exists(path):
        return None, None

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid CT artifact at {path}: expected a dict payload.")

    if "features" not in data:
        raise ValueError(
            "Legacy CT artifact format detected. "
            "Retrain artifacts with the current train.py before submitting."
        )

    model = data.get("model")
    feature_bundle = data.get("features")
    model_class = type(model).__name__ if model is not None else None
    if model_class not in SUPPORTED_CT_MODEL_TYPES:
        raise ValueError(
            f"Unsupported CT model type '{model_class}'. "
            f"Expected one of: {sorted(SUPPORTED_CT_MODEL_TYPES)}."
        )

    if not isinstance(feature_bundle, dict):
        raise ValueError("Invalid CT artifact: 'features' must be a dict.")

    required_keys = {"char_tfidf", "word_tfidf"}
    missing_keys = required_keys - set(feature_bundle)
    if missing_keys:
        raise ValueError(
            f"Invalid CT artifact: missing feature extractors {sorted(missing_keys)}."
        )

    return model, feature_bundle


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

        normalized_queries = [normalize(query, use_lemmatization=True) for query in queries]
        model_class = type(self.model).__name__
        if model_class not in SUPPORTED_CT_MODEL_TYPES:
            raise ValueError(
                f"Unsupported CT model type '{model_class}' at inference time. "
                "Please retrain submission artifacts."
            )

        feature_matrix = _transform_texts(normalized_queries, self.feature_bundle)
        preds = self.model.predict(feature_matrix)
        normalized_preds = [pred if pred in self.VALID_LABELS else "прочее" for pred in preds]

        # Calculate margins
        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(feature_matrix)
            if getattr(decision, "ndim", 1) == 1:
                margins = [abs(float(score)) for score in decision]
            else:
                margins = []
                for row in decision:
                    ranked = np.sort(np.asarray(row, dtype=float))
                    margins.append(float(ranked[-1] - ranked[-2]) if len(ranked) >= 2 else abs(float(ranked[0])) if len(ranked) == 1 else None)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_matrix)
            margins = []
            for row in probs:
                ranked = np.sort(np.asarray(row, dtype=float))
                margins.append(float(ranked[-1] - ranked[-2]) if len(ranked) >= 2 else float(ranked[0]))
        else:
            return [(pred, None) for pred in normalized_preds]

        return list(zip(normalized_preds, margins))
