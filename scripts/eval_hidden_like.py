"""Run hidden-like local evaluation for the Mediascope pipeline.

The script trains artifacts on a train-fold, reloads the same pipeline logic
used in production, and evaluates it on validation folds that better imitate
leaderboard conditions than a full-train self-check.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
LIBS = ROOT / "libs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if LIBS.is_dir() and str(LIBS) not in sys.path:
    sys.path.insert(0, str(LIBS))

from aggregator import aggregate_predictions
from ct_classifier import (
    ContentTypeClassifier,
    load_artifacts as load_ct,
    save_artifacts as save_ct,
    train_ct_classifier,
)
from ct_calibration import calibrate_content_type, detect_ct_from_words
from embeddings import (
    EmbeddingMatcher,
    build_embedding_index,
    load_artifacts as load_embeddings,
    save_artifacts as save_embeddings,
)
from franchise_dict import (
    FranchiseMatcher,
    build_franchise_dict,
    load_artifacts as load_franchise,
    save_artifacts as save_franchise,
)
from knowledge_graph import (
    KnowledgeGraphMatcher,
    build_knowledge_graph,
    load_artifacts as load_knowledge_graph,
    save_artifacts as save_knowledge_graph,
)
from preprocessing import detect_translit, load_artifacts as load_preprocessing
from preprocessing import save_artifacts as save_preprocessing
from preprocessing import transliterate
from preprocessing import build_typo_dict, normalize
from supplemental_titles import load_kinopoisk_top250, merge_title_dicts
from title_retrieval import TitleRetriever
from typequery_classifier import (
    TypeQueryClassifier,
    load_artifacts as load_typequery,
    save_artifacts as save_typequery,
    train_typequery_classifier,
)

CONTENT_TYPE_LABELS = ["сериал", "фильм", "мультфильм", "мультсериал", "прочее"]
CONTENT_TYPE_MAP = {label: label for label in CONTENT_TYPE_LABELS}


def _map_content_type(raw: str) -> str:
    """Normalize content type to the expected leaderboard label space."""
    if not raw:
        return "прочее"
    return CONTENT_TYPE_MAP.get(raw.strip().lower(), "прочее")
@dataclass
class EvalMetrics:
    """Leaderboard-style metrics for a single validation fold."""

    typequery_f2: float
    contenttype_macro_f1: float
    title_token_f1: float
    combined_score: float
    false_positive_count: int
    false_negative_count: int
    exact_title_match_count: int
    positive_query_count: int


class LocalPredictionModel:
    """Artifact-backed copy of the production inference pipeline."""

    batch_size: int = 10

    def __init__(self, artifact_dir: Path, repo_root: Path) -> None:
        """Load all pre-built artifacts from the provided directory."""
        self._artifact_dir = artifact_dir
        self._repo_root = repo_root

        meta_path = artifact_dir / "metadata.json"
        if meta_path.exists():
            self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self._metadata = {}

        load_preprocessing(str(artifact_dir / "typo_dict.json"))

        tq_model, tq_tfidf, tq_threshold = load_typequery(str(artifact_dir / "typequery_model.pkl"))
        self._tq_classifier = (
            TypeQueryClassifier(tq_model, tq_tfidf, tq_threshold) if tq_model is not None else None
        )

        franchise_dict = load_franchise(str(artifact_dir / "franchise_dict.json"))
        supplemental_titles = load_kinopoisk_top250(repo_root / "kinopoisk-top250.csv")
        franchise_dict = merge_title_dicts(franchise_dict, supplemental_titles)
        self._franchise_matcher = FranchiseMatcher(franchise_dict) if franchise_dict else None
        self._title_retriever = TitleRetriever(franchise_dict) if franchise_dict else None

        vectorizer, prototypes, prototype_info = load_embeddings(str(artifact_dir / "embeddings.pkl"))
        self._embedding_matcher = (
            EmbeddingMatcher(vectorizer, prototypes, prototype_info) if vectorizer is not None else None
        )

        edges, node_types, ct_map = load_knowledge_graph(str(artifact_dir / "knowledge_graph.json"))
        self._graph_matcher = KnowledgeGraphMatcher(edges, node_types, ct_map) if edges else None

        ct_model, ct_tfidf = load_ct(str(artifact_dir / "ct_classifier.pkl"))
        self._ct_classifier = (
            ContentTypeClassifier(ct_model, ct_tfidf) if ct_model is not None else None
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the same end-to-end inference logic as the submission model."""
        queries = df["QueryText"].tolist()
        out = pd.DataFrame(
            {
                "QueryText": queries,
                "TypeQuery": 0,
                "Title": "",
                "ContentType": "прочее",
            }
        )

        if self._tq_classifier is not None:
            tq_preds, _ = self._tq_classifier.predict(queries)
        else:
            tq_preds = [self._heuristic_typequery(query) for query in queries]

        out["TypeQuery"] = tq_preds

        positive_indices = [index for index, pred in enumerate(tq_preds) if pred == 1]
        if not positive_indices:
            return out

        positive_queries = [queries[index] for index in positive_indices]
        ct_preds = (
            self._ct_classifier.predict_with_margins(positive_queries)
            if self._ct_classifier is not None
            else None
        )

        for batch_offset, index in enumerate(positive_indices):
            query = positive_queries[batch_offset]
            query_cyr = transliterate(query) if detect_translit(query) else query

            retrieval = self._title_retriever.retrieve(query_cyr) if self._title_retriever is not None else None
            raw_candidate = retrieval.raw_candidate if retrieval is not None else ""
            is_generic = (
                self._embedding_matcher._is_generic_query(query_cyr) if self._embedding_matcher is not None else False
            )

            fm_result = self._franchise_matcher.match(query_cyr) if self._franchise_matcher else ("", "", 0.0, None)
            em_results = self._embedding_matcher.match(query_cyr) if self._embedding_matcher else []
            gm_results = self._graph_matcher.match(query_cyr) if self._graph_matcher else []
            if is_generic:
                gm_results = []

            title = ""
            title_content_type = ""
            title_source = ""

            if fm_result[3] == "exact" and len(fm_result[0]) >= 2 and not fm_result[0].isdigit():
                title = fm_result[0]
                title_content_type = fm_result[1]
                title_source = "franchise_exact"
            elif retrieval and retrieval.title and retrieval.confidence >= 0.84:
                title = retrieval.title
                title_content_type = retrieval.content_type
                title_source = retrieval.source
            elif fm_result[3] == "substring" and fm_result[2] > 0.5:
                if len(fm_result[0]) >= 2 and not fm_result[0].isdigit():
                    title_words = set(fm_result[0].split())
                    query_words = set(normalize(query_cyr).split())
                    overlap = len(title_words & query_words) / max(len(title_words), 1)
                    if overlap >= 0.5 or fm_result[2] > 0.75:
                        title = fm_result[0]
                        title_content_type = fm_result[1]
                        title_source = "franchise_substring"
            elif (fm_result[0] or em_results or gm_results) and not is_generic:
                agg = aggregate_predictions(fm_result, em_results, gm_results)
                if agg["title"] and agg["confidence"] > 0.15 and agg["agreement"] > 0.0:
                    is_consistent = not raw_candidate
                    if retrieval and raw_candidate:
                        is_consistent = self._title_retriever.is_match_consistent(raw_candidate, agg["title"])
                    is_query_compatible = self._title_retriever.is_query_compatible(query_cyr, agg["title"])
                    if (
                        is_consistent
                        and is_query_compatible
                        and len(agg["title"]) >= 2
                        and not agg["title"].isdigit()
                    ):
                        title = agg["title"]
                        title_content_type = agg["contentType"]
                        title_source = "aggregate"

            if not title and fm_result[3] == "fuzzy" and fm_result[2] > 0.6 and not is_generic:
                if len(fm_result[0]) >= 3 and not fm_result[0].isdigit():
                    is_consistent = not raw_candidate
                    if retrieval and raw_candidate:
                        is_consistent = self._title_retriever.is_match_consistent(raw_candidate, fm_result[0])
                    is_query_compatible = self._title_retriever.is_query_compatible(query_cyr, fm_result[0])
                    if (
                        is_consistent
                        and is_query_compatible
                        and (len(fm_result[0]) >= 5 or fm_result[2] >= 0.75)
                    ):
                        title = fm_result[0]
                        title_content_type = fm_result[1]
                        title_source = "franchise_fuzzy"

            if (
                not title
                and retrieval
                and raw_candidate
                and self._title_retriever.should_accept_raw_candidate(query_cyr, raw_candidate)
            ):
                title = raw_candidate
                title_content_type = retrieval.content_type
                title_source = retrieval.source or "raw"

            model_content_type = ""
            model_margin = None
            if ct_preds:
                model_content_type, model_margin = ct_preds[batch_offset]

            out.at[index, "Title"] = title
            calibrated_ct = calibrate_content_type(
                query=query,
                title_content_type=title_content_type,
                model_content_type=model_content_type,
                model_margin=model_margin,
                title_source=title_source,
            )
            out.at[index, "ContentType"] = _map_content_type(calibrated_ct or detect_ct_from_words(query))

        return out

    @staticmethod
    def _heuristic_typequery(query: str) -> int:
        """Fallback TypeQuery heuristic for artifact-free runs."""
        lower = query.lower()
        video_signals = [
            "смотреть",
            "скачать",
            "онлайн",
            "торрент",
            "сериал",
            "фильм",
            "аниме",
            "дорама",
            "мульт",
            "озвучк",
            "сезон",
            "серия",
            "hd",
            "1080",
            "720",
        ]
        non_video_signals = [
            "купить",
            "цена",
            "рецепт",
            "приготов",
            "скачать apk",
            "взлом",
            "погод",
            "расписани",
        ]
        video_score = sum(1 for marker in video_signals if marker in lower)
        non_video_score = sum(1 for marker in non_video_signals if marker in lower)
        return 1 if video_score > non_video_score else 0


def build_validation_splits(
    df: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create evaluation splits with different difficulty profiles."""
    train_idx, valid_idx = train_test_split(
        df.index.to_numpy(),
        test_size=validation_fraction,
        stratify=df["TypeQuery"],
        random_state=random_state,
    )
    splits = {
        "random_stratified": (df.loc[train_idx].copy(), df.loc[valid_idx].copy()),
    }

    title_mask = (
        (df["TypeQuery"] == 1)
        & df["Title"].notna()
        & df["Title"].astype(str).str.strip().ne("")
    )
    positive_with_title = df.loc[title_mask].copy()
    positive_with_title["title_group"] = positive_with_title["Title"].astype(str).str.strip().str.lower()

    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    grouped_train_pos, grouped_valid_pos = next(
        group_splitter.split(
            positive_with_title,
            groups=positive_with_title["title_group"],
        )
    )

    train_indices = set(positive_with_title.iloc[grouped_train_pos].index)
    valid_indices = set(positive_with_title.iloc[grouped_valid_pos].index)

    remainder = df.loc[~title_mask].copy()
    remainder_train_idx, remainder_valid_idx = train_test_split(
        remainder.index.to_numpy(),
        test_size=validation_fraction,
        stratify=remainder["TypeQuery"],
        random_state=random_state,
    )
    train_indices.update(remainder_train_idx.tolist())
    valid_indices.update(remainder_valid_idx.tolist())

    splits["title_group_holdout"] = (
        df.loc[sorted(train_indices)].copy(),
        df.loc[sorted(valid_indices)].copy(),
    )
    return splits


def subsample_validation_df(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Return a stratified validation subset when a faster proxy run is needed."""
    if max_rows is None or len(df) <= max_rows:
        return df

    sample_df = df.copy()
    sample_df["sample_group"] = np.where(
        sample_df["TypeQuery"].astype(int) == 1,
        "1|" + sample_df["ContentType"].fillna("прочее").astype(str),
        "0|none",
    )
    sampled_idx, _ = train_test_split(
        sample_df.index.to_numpy(),
        train_size=max_rows,
        stratify=sample_df["sample_group"],
        random_state=random_state,
    )
    return df.loc[sorted(sampled_idx)].copy()


def train_artifacts(train_df: pd.DataFrame, artifact_dir: Path, repo_root: Path) -> None:
    """Train all artifacts needed by the local pipeline into a temp directory."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    franchise_dict = build_franchise_dict(train_df)
    supplemental_titles = load_kinopoisk_top250(repo_root / "kinopoisk-top250.csv")
    if supplemental_titles:
        franchise_dict = merge_title_dicts(franchise_dict, supplemental_titles)
    save_franchise(franchise_dict, str(artifact_dir / "franchise_dict.json"))

    canonical_titles = list(franchise_dict.keys())
    typo_dict = build_typo_dict(train_df["QueryText"].tolist(), canonical_titles)
    save_preprocessing(typo_dict, str(artifact_dir / "typo_dict.json"))

    tq_model, tq_tfidf, tq_threshold = train_typequery_classifier(train_df)
    save_typequery(tq_model, tq_tfidf, tq_threshold, str(artifact_dir / "typequery_model.pkl"))

    vectorizer, prototypes, prototype_info = build_embedding_index(train_df)
    save_embeddings(vectorizer, prototypes, prototype_info, str(artifact_dir / "embeddings.pkl"))

    edges, node_types = build_knowledge_graph(train_df)
    content_type_map = {
        title: data["contentType"]
        for title, data in franchise_dict.items()
    }
    save_knowledge_graph(
        edges,
        node_types,
        content_type_map,
        str(artifact_dir / "knowledge_graph.json"),
    )

    ct_model, ct_tfidf = train_ct_classifier(train_df)
    save_ct(ct_model, ct_tfidf, str(artifact_dir / "ct_classifier.pkl"))

    metadata = {
        "n_train": int(len(train_df)),
        "n_titles": int(len(franchise_dict)),
        "n_supplemental_titles": int(len(supplemental_titles)),
        "typequery_threshold": float(tq_threshold),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_proxy_typequery_classifier(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[LogisticRegression, TfidfVectorizer, float]:
    """Train a lighter TypeQuery proxy for fast hidden-like iteration loops."""
    from scipy.sparse import csr_matrix, hstack
    from preprocessing import extract_features, features_to_vector

    x_text = df["QueryText"].tolist()
    y = df["TypeQuery"].astype(int).to_numpy()

    x_train_text, x_val_text, y_train, y_val = train_test_split(
        x_text,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    tfidf = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        analyzer="char_wb",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    x_train_tfidf = tfidf.fit_transform(x_train_text)
    x_val_tfidf = tfidf.transform(x_val_text)

    x_train_hand = np.array([features_to_vector(extract_features(text)) for text in x_train_text])
    x_val_hand = np.array([features_to_vector(extract_features(text)) for text in x_val_text])

    x_train = hstack([x_train_tfidf, csr_matrix(x_train_hand)])
    x_val = hstack([x_val_tfidf, csr_matrix(x_val_hand)])

    model = LogisticRegression(
        max_iter=1200,
        solver="liblinear",
        class_weight={0: 1.0, 1: 1.8},
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    best_threshold = 0.5
    best_f2 = -1.0
    for threshold in np.arange(0.1, 0.91, 0.01):
        preds = (val_probs >= threshold).astype(int)
        score = fbeta_score(y_val, preds, beta=2)
        if score > best_f2:
            best_f2 = score
            best_threshold = float(threshold)

    x_full_tfidf = tfidf.transform(x_text)
    x_full_hand = np.array([features_to_vector(extract_features(text)) for text in x_text])
    x_full = hstack([x_full_tfidf, csr_matrix(x_full_hand)])
    model.fit(x_full, y)

    return model, tfidf, best_threshold


def train_artifacts_fast_proxy(train_df: pd.DataFrame, artifact_dir: Path, repo_root: Path) -> None:
    """Train a faster proxy set of artifacts for daily hidden-like loops."""
    started_at = time.perf_counter()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("  [fast_proxy] build franchise_dict", flush=True)
    franchise_dict = build_franchise_dict(train_df)
    supplemental_titles = load_kinopoisk_top250(repo_root / "kinopoisk-top250.csv")
    if supplemental_titles:
        franchise_dict = merge_title_dicts(franchise_dict, supplemental_titles)
    save_franchise(franchise_dict, str(artifact_dir / "franchise_dict.json"))

    print("  [fast_proxy] build typo_dict", flush=True)
    canonical_titles = list(franchise_dict.keys())
    typo_dict = build_typo_dict(train_df["QueryText"].tolist(), canonical_titles)
    save_preprocessing(typo_dict, str(artifact_dir / "typo_dict.json"))

    print("  [fast_proxy] train typequery proxy", flush=True)
    tq_model, tq_tfidf, tq_threshold = train_proxy_typequery_classifier(train_df)
    save_typequery(tq_model, tq_tfidf, tq_threshold, str(artifact_dir / "typequery_model.pkl"))

    print("  [fast_proxy] build embeddings", flush=True)
    vectorizer, prototypes, prototype_info = build_embedding_index(train_df)
    save_embeddings(vectorizer, prototypes, prototype_info, str(artifact_dir / "embeddings.pkl"))

    print("  [fast_proxy] build knowledge graph", flush=True)
    edges, node_types = build_knowledge_graph(train_df)
    content_type_map = {
        title: data["contentType"]
        for title, data in franchise_dict.items()
    }
    save_knowledge_graph(
        edges,
        node_types,
        content_type_map,
        str(artifact_dir / "knowledge_graph.json"),
    )

    print("  [fast_proxy] train contenttype proxy", flush=True)
    ct_model, ct_features = train_ct_classifier(train_df)
    save_ct(ct_model, ct_features, str(artifact_dir / "ct_classifier.pkl"))

    metadata = {
        "n_train": int(len(train_df)),
        "n_titles": int(len(franchise_dict)),
        "n_supplemental_titles": int(len(supplemental_titles)),
        "training_mode": "fast_proxy",
        "typequery_threshold": float(tq_threshold),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started_at
    print(f"  [fast_proxy] artifacts ready in {elapsed:.1f}s", flush=True)


def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> EvalMetrics:
    """Calculate leaderboard-style metrics on a validation fold."""
    true_typequery = y_true["TypeQuery"].astype(int).to_numpy()
    pred_typequery = y_pred["TypeQuery"].astype(int).to_numpy()
    typequery_f2 = fbeta_score(true_typequery, pred_typequery, beta=2)

    positive_mask = y_true["TypeQuery"].astype(int) == 1
    ct_true = y_true.loc[positive_mask, "ContentType"].fillna("прочее").astype(str).str.lower().to_numpy()
    ct_pred = y_pred.loc[positive_mask, "ContentType"].fillna("прочее").astype(str).str.lower().to_numpy()
    contenttype_macro_f1 = f1_score(
        ct_true,
        ct_pred,
        average="macro",
        labels=CONTENT_TYPE_LABELS,
        zero_division=0,
    )

    title_scores = []
    exact_title_match_count = 0
    for index in y_true.index[positive_mask]:
        true_tokens = set(str(y_true.at[index, "Title"]).lower().split())
        pred_tokens = set(str(y_pred.at[index, "Title"]).lower().split())
        if true_tokens == pred_tokens and true_tokens:
            exact_title_match_count += 1
        if not true_tokens and not pred_tokens:
            title_scores.append(1.0)
            continue
        if not true_tokens or not pred_tokens:
            title_scores.append(0.0)
            continue
        overlap = true_tokens & pred_tokens
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(true_tokens)
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        title_scores.append(score)

    title_token_f1 = float(np.mean(title_scores)) if title_scores else 0.0
    combined_score = 0.35 * typequery_f2 + 0.30 * contenttype_macro_f1 + 0.35 * title_token_f1

    false_positive_count = int(((true_typequery == 0) & (pred_typequery == 1)).sum())
    false_negative_count = int(((true_typequery == 1) & (pred_typequery == 0)).sum())

    return EvalMetrics(
        typequery_f2=float(typequery_f2),
        contenttype_macro_f1=float(contenttype_macro_f1),
        title_token_f1=float(title_token_f1),
        combined_score=float(combined_score),
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        exact_title_match_count=exact_title_match_count,
        positive_query_count=int(positive_mask.sum()),
    )


def evaluate_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    repo_root: Path,
    mode: str,
) -> EvalMetrics:
    """Train on one fold and evaluate it with the production-like pipeline."""
    with tempfile.TemporaryDirectory(prefix="hidden_like_", dir="/tmp") as temp_dir:
        artifact_dir = Path(temp_dir) / "artifacts"
        started_at = time.perf_counter()
        if mode == "full":
            train_artifacts(train_df, artifact_dir, repo_root)
        else:
            train_artifacts_fast_proxy(train_df, artifact_dir, repo_root)
        train_elapsed = time.perf_counter() - started_at
        print(f"  [eval] training finished in {train_elapsed:.1f}s", flush=True)

        inference_started_at = time.perf_counter()
        model = LocalPredictionModel(artifact_dir, repo_root)
        predictions = model.predict(valid_df[["QueryText"]].copy())
        inference_elapsed = time.perf_counter() - inference_started_at
        print(f"  [eval] inference finished in {inference_elapsed:.1f}s", flush=True)
        aligned_predictions = predictions.set_index(valid_df.index)
        return compute_metrics(valid_df, aligned_predictions)


def format_metrics(metrics: EvalMetrics) -> str:
    """Format metrics for readable CLI output."""
    return (
        f"score={metrics.combined_score:.4f} | "
        f"tq_f2={metrics.typequery_f2:.4f} | "
        f"ct_f1={metrics.contenttype_macro_f1:.4f} | "
        f"title_f1={metrics.title_token_f1:.4f} | "
        f"fp={metrics.false_positive_count} | "
        f"fn={metrics.false_negative_count} | "
        f"exact_title={metrics.exact_title_match_count}/{metrics.positive_query_count}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "train.csv",
        help="Path to the labeled train.csv file.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation share used for each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for all splits.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "random_stratified", "title_group_holdout"],
        default="all",
        help="Which hidden-like split to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["fast_proxy", "full"],
        default="fast_proxy",
        help="Training mode for the evaluation contour.",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=None,
        help="Optional cap for a stratified validation subset in faster proxy runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Train and evaluate the pipeline on hidden-like validation splits."""
    args = parse_args()
    df = pd.read_csv(args.train_path)
    splits = build_validation_splits(df, args.validation_fraction, args.seed)

    selected_names = [args.split] if args.split != "all" else list(splits.keys())
    all_metrics: dict[str, EvalMetrics] = {}

    for split_name in selected_names:
        train_df, valid_df = splits[split_name]
        valid_df = subsample_validation_df(valid_df, args.max_valid_rows, args.seed)
        print(
            f"[{split_name}] train={len(train_df)} rows | "
            f"valid={len(valid_df)} rows | mode={args.mode}",
            flush=True,
        )
        metrics = evaluate_split(train_df, valid_df, ROOT, args.mode)
        all_metrics[split_name] = metrics
        print(format_metrics(metrics), flush=True)

    if len(all_metrics) > 1:
        avg_metrics = EvalMetrics(
            **{
                field: float(np.mean([getattr(metrics, field) for metrics in all_metrics.values()]))
                for field in EvalMetrics.__dataclass_fields__
            }
        )
        print("[average] " + format_metrics(avg_metrics), flush=True)

    print("\nJSON summary:")
    print(
        json.dumps(
            {name: asdict(metrics) for name, metrics in all_metrics.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
