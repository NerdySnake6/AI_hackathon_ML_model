"""
ContentType classifier: predicts ContentType for TypeQuery=1 queries.
Used as fallback when title-based CT is unavailable.
"""
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocessing import normalize, extract_features, features_to_vector


def train_ct_classifier(df):
    """
    Train ContentType classifier on TypeQuery=1 queries.
    Returns (classifier, tfidf_vectorizer).
    """
    pos = df[df['TypeQuery'] == 1].copy()
    pos['ContentType'] = pos['ContentType'].fillna('').astype(str)
    pos.loc[pos['ContentType'] == '', 'ContentType'] = 'прочее'

    X_text = pos['QueryText'].tolist()
    y = pos['ContentType'].values

    print(f"  Training on {len(pos)} positive queries")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # TF-IDF features — lighter config for speed
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        analyzer='char_wb',
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # Train on full data — simpler model for speed
    final_model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        l2_regularization=5.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    final_model.fit(X_tfidf.toarray(), y)

    # Quick evaluation on validation split
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X_tfidf, y, test_size=0.1, stratify=y, random_state=42)
    preds = final_model.predict(X_val.toarray())
    cv_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
    print(f"  Holdout macro F1: {cv_f1:.4f}")

    return final_model, tfidf


def save_artifacts(model, tfidf, path: str = "artifacts/ct_classifier.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'tfidf': tfidf}, f)


def load_artifacts(path: str = "artifacts/ct_classifier.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['tfidf']
    return None, None


class ContentTypeClassifier:
    """Predict ContentType for TypeQuery=1 queries."""

    VALID_LABELS = ['сериал', 'фильм', 'мультфильм', 'мультсериал', 'прочее']

    def __init__(self, model, tfidf):
        self.model = model
        self.tfidf = tfidf

    def predict(self, queries: list) -> list:
        """Predict ContentType for a list of queries."""
        if self.model is None or self.tfidf is None:
            return ['прочее'] * len(queries)

        X_tfidf = self.tfidf.transform(queries)
        preds = self.model.predict(X_tfidf.toarray())
        return [p if p in self.VALID_LABELS else 'прочее' for p in preds]
