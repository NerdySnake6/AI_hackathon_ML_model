"""
TypeQuery binary classifier: gradient boosting on TF-IDF + hand-crafted features.
Optimized for F2 score (recall-heavy).
"""
import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from preprocessing import normalize, extract_features, features_to_vector, FEATURE_KEYS


def train_typequery_classifier(df, random_state: int = 42):
    """
    Train a TypeQuery classifier combining TF-IDF and hand-crafted features.
    Returns (model, tfidf_vectorizer, feature_scaler, threshold).
    """
    X_text = df['QueryText'].tolist()
    y = df['TypeQuery'].values

    # Split for threshold tuning
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.15, stratify=y, random_state=random_state
    )

    # TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        analyzer='char_wb',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)

    # Hand-crafted features
    X_train_hand = np.array([features_to_vector(extract_features(t)) for t in X_train_text])
    X_val_hand = np.array([features_to_vector(extract_features(t)) for t in X_val_text])

    # Combine
    from scipy.sparse import hstack, csr_matrix
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_hand)])
    X_val = hstack([X_val_tfidf, csr_matrix(X_val_hand)])

    # HistGradientBoosting needs dense arrays
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()

    # Train classifier
    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train_dense, y_train)

    # Find optimal threshold for F2
    val_probs = model.predict_proba(X_val_dense)[:, 1]
    best_f2 = 0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (val_probs >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    print(f"TypeQuery F2 on val: {best_f2:.4f} (threshold={best_threshold:.2f})")

    # Retrain on full data with found threshold
    X_full_tfidf = tfidf.transform(X_text)
    X_full_hand = np.array([features_to_vector(extract_features(t)) for t in X_text])
    from scipy.sparse import hstack
    X_full = hstack([X_full_tfidf, csr_matrix(X_full_hand)])
    model.fit(X_full.toarray(), y)

    return model, tfidf, best_threshold


def save_artifacts(model, tfidf, threshold, path: str = "artifacts/typequery_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'tfidf': tfidf,
            'threshold': threshold,
        }, f)


def load_artifacts(path: str = "artifacts/typequery_model.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['tfidf'], data['threshold']
    return None, None, 0.5


class TypeQueryClassifier:
    """Predict TypeQuery (0/1) for a batch of queries."""

    def __init__(self, model, tfidf, threshold):
        self.model = model
        self.tfidf = tfidf
        self.threshold = threshold

    def predict(self, queries: list) -> tuple:
        """
        Predict TypeQuery for a list of queries.
        Returns (predictions: list[int], probabilities: list[float]).
        """
        tfidf_features = self.tfidf.transform(queries)
        hand_features = np.array([features_to_vector(extract_features(t)) for t in queries])

        from scipy.sparse import hstack, csr_matrix
        X = hstack([tfidf_features, csr_matrix(hand_features)])
        X_dense = X.toarray()

        probs = self.model.predict_proba(X_dense)[:, 1]
        predictions = (probs >= self.threshold).astype(int)

        return predictions.tolist(), probs.tolist()
