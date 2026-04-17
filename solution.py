"""
Solution entry point for the Mediascope hackathon.
Implements PredictionModel with a cascading pipeline:
  1. TypeQuery binary classifier
  2. For TypeQuery=1: parallel 3-branch title detection
     - Fuzzy dictionary matching (relaxed thresholds)
     - TF-IDF embedding prototype matching
     - Knowledge graph traversal
  3. Meta-aggregator combines branch results
  4. Fallbacks: CT classifier + title extraction
"""
import sys
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Bundled dependencies (libs/ directory inside the bundle)
_LIBS = _ROOT / "libs"
if _LIBS.is_dir() and str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))

from preprocessing import normalize, load_artifacts as load_preprocessing
from preprocessing import detect_translit, transliterate
from franchise_dict import load_artifacts as load_franchise, FranchiseMatcher
from typequery_classifier import load_artifacts as load_typequery, TypeQueryClassifier
from embeddings import load_artifacts as load_embeddings, EmbeddingMatcher
from knowledge_graph import load_artifacts as load_knowledge_graph, KnowledgeGraphMatcher
from aggregator import aggregate_predictions
from ct_classifier import load_artifacts as load_ct, ContentTypeClassifier
from title_extraction import extract_title_candidate
from supplemental_titles import load_kinopoisk_top250, merge_title_dicts

# ---------------------------------------------------------------------------
# ContentType mapping – align our predictions with expected label space
# ---------------------------------------------------------------------------
CONTENT_TYPE_MAP = {
    'сериал': 'сериал',
    'фильм': 'фильм',
    'мультфильм': 'мультфильм',
    'мультсериал': 'мультсериал',
    'прочее': 'прочее',
}


def _map_content_type(raw: str) -> str:
    """Normalize content type to the expected label set."""
    if not raw:
        return 'прочее'
    raw_lower = raw.strip().lower()
    return CONTENT_TYPE_MAP.get(raw_lower, 'прочее')


def _detect_ct_from_words(query: str) -> str:
    """Detect ContentType from keyword hints in the query."""
    lower = query.lower()
    if any(w in lower for w in ['мультфильм', 'мульт ф', 'cartoon', 'анимацион']):
        return 'мультфильм'
    if any(w in lower for w in ['мультсериал', 'мульт сериал']):
        return 'мультсериал'
    if any(w in lower for w in ['сериал', 'дорам', 'сезон', 'серия']):
        return 'сериал'
    if any(w in lower for w in ['фильм', 'фильмы', 'картина', 'кино']):
        return 'фильм'
    return ''


# ---------------------------------------------------------------------------
# PredictionModel – the only class the sandbox expects
# ---------------------------------------------------------------------------
class PredictionModel:
    """Hackathon inference pipeline used by the evaluation sandbox."""

    batch_size: int = 10  # sandbox will call predict() in batches of this size

    def __init__(self) -> None:
        """Load all pre-built artifacts at startup."""
        art_dir = _ROOT / 'artifacts'

        # Metadata
        meta_path = art_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

        # Typo dictionary
        load_preprocessing(str(art_dir / 'typo_dict.json'))

        # TypeQuery classifier
        model, tfidf, threshold = load_typequery(str(art_dir / 'typequery_model.pkl'))
        self._tq_classifier = TypeQueryClassifier(model, tfidf, threshold) if model else None

        # Franchise dictionary
        franchise_dict = load_franchise(str(art_dir / 'franchise_dict.json'))
        supplemental_titles = load_kinopoisk_top250(_ROOT / 'kinopoisk-top250.csv')
        franchise_dict = merge_title_dicts(franchise_dict, supplemental_titles)
        self._franchise_matcher = FranchiseMatcher(franchise_dict) if franchise_dict else None

        # Embedding index
        vectorizer, prototypes, prototype_info = load_embeddings(str(art_dir / 'embeddings.pkl'))
        self._embedding_matcher = EmbeddingMatcher(vectorizer, prototypes, prototype_info) if vectorizer else None

        # Knowledge graph
        edges, node_types, ct_map = load_knowledge_graph(str(art_dir / 'knowledge_graph.json'))
        self._graph_matcher = KnowledgeGraphMatcher(edges, node_types, ct_map) if edges else None

        # ContentType classifier (fallback)
        ct_model, ct_tfidf = load_ct(str(art_dir / 'ct_classifier.pkl'))
        self._ct_classifier = ContentTypeClassifier(ct_model, ct_tfidf) if ct_model else None

        # Default CT
        self._default_ct = 'прочее'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain column 'QueryText' with raw query strings.

        Returns
        -------
        pd.DataFrame with columns: QueryText, TypeQuery (int 0/1),
        Title (str), ContentType (str)
        """
        queries = df['QueryText'].tolist()

        # Prepare output
        out = pd.DataFrame({
            'QueryText': queries,
            'TypeQuery': 0,
            'Title': '',
            'ContentType': 'прочее',
        })

        # Step 1: TypeQuery classification
        if self._tq_classifier is not None:
            tq_preds, tq_probs = self._tq_classifier.predict(queries)
        else:
            tq_preds = [self._heuristic_typequery(q) for q in queries]
            tq_probs = [1.0 if p else 0.0 for p in tq_preds]

        out['TypeQuery'] = tq_preds

        # Step 2: For TypeQuery=1, run 3-branch title detection
        positive_indices = [i for i, p in enumerate(tq_preds) if p == 1]
        if not positive_indices:
            return out

        positive_queries = [queries[i] for i in positive_indices]

        # Pre-compute CT classifier predictions for all positive queries
        if self._ct_classifier is not None:
            ct_preds = self._ct_classifier.predict(positive_queries)
        else:
            ct_preds = None

        for batch_offset in range(len(positive_indices)):
            idx = positive_indices[batch_offset]
            query = positive_queries[batch_offset]

            # Detect transliteration
            query_cyr = query
            if detect_translit(query):
                query_cyr = transliterate(query)

            # Check if this is a generic query (no specific title expected)
            is_generic = self._embedding_matcher._is_generic_query(query_cyr) if self._embedding_matcher else False

            # Branch 1: Franchise dictionary
            if self._franchise_matcher:
                fm_result = self._franchise_matcher.match(query_cyr)
            else:
                fm_result = ('', '', 0.0, None)

            # Branch 2: Embedding matching
            if self._embedding_matcher:
                em_results = self._embedding_matcher.match(query_cyr)
            else:
                em_results = []

            # Branch 3: Knowledge graph
            if self._graph_matcher:
                gm_results = self._graph_matcher.match(query_cyr)
                # Filter out graph matches for generic queries
                if is_generic:
                    gm_results = []
            else:
                gm_results = []

            # ---- Determine title and content type ----
            title = ''
            content_type = ''

            # Strategy A: Exact franchise match (highest priority)
            if fm_result[3] == 'exact' and len(fm_result[0]) >= 2 and not fm_result[0].isdigit():
                title = fm_result[0]
                content_type = fm_result[1]

            # Strategy B: Substring match with overlap check
            elif fm_result[3] == 'substring' and fm_result[2] > 0.5:
                if len(fm_result[0]) >= 2 and not fm_result[0].isdigit():
                    title_words = set(fm_result[0].split())
                    query_words = set(normalize(query_cyr).split())
                    overlap = len(title_words & query_words) / max(len(title_words), 1)
                    if overlap >= 0.5 or fm_result[2] > 0.75:
                        title = fm_result[0]
                        content_type = fm_result[1]

            # Strategy C: Aggregated result from all branches — skip for generic
            elif (fm_result[0] or em_results or gm_results) and not is_generic:
                agg = aggregate_predictions(fm_result, em_results, gm_results)
                if agg['title'] and agg['confidence'] > 0.15 and agg['agreement'] > 0.0:
                    if len(agg['title']) >= 2 and not agg['title'].isdigit():
                        title = agg['title']
                        content_type = agg['contentType']

            # Strategy D: Partial franchise match — skip for generic queries
            if not title and fm_result[3] == 'fuzzy' and fm_result[2] > 0.6 and not is_generic:
                # Extra protection: reject very short titles and generic queries
                if len(fm_result[0]) >= 3 and not fm_result[0].isdigit():
                    # For short titles, require higher confidence
                    if len(fm_result[0]) < 5 and fm_result[2] < 0.75:
                        pass  # skip
                    else:
                        title = fm_result[0]
                        content_type = fm_result[1]

            # Strategy E: Title extraction from query — skip for generic queries
            if not title and not is_generic:
                extracted = extract_title_candidate(query_cyr)
                if extracted and len(extracted) >= 2:
                    title = extracted
                    # Try to infer CT from the full query
                    content_type = _detect_ct_from_words(query)

            # ---- Determine content type ----
            if not content_type:
                # Use CT classifier if available
                if ct_preds and ct_preds[batch_offset]:
                    content_type = ct_preds[batch_offset]
                else:
                    content_type = _detect_ct_from_words(query)

            out.at[idx, 'Title'] = title
            out.at[idx, 'ContentType'] = _map_content_type(content_type)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_typequery(query: str) -> int:
        """Fallback heuristic for TypeQuery when model is unavailable."""
        lower = query.lower()
        video_signals = [
            'смотреть', 'скачать', 'онлайн', 'торрент', 'сериал',
            'фильм', 'аниме', 'дорама', 'мульт', 'озвучк', 'сезон',
            'серия', 'hd', '1080', '720',
        ]
        non_video_signals = [
            'купить', 'цена', 'рецепт', 'приготов', 'скачать apk',
            'взлом', 'погод', 'расписани',
        ]
        video_score = sum(1 for m in video_signals if m in lower)
        non_video_score = sum(1 for m in non_video_signals if m in lower)
        return 1 if video_score > non_video_score else 0


# ---------------------------------------------------------------------------
# Standalone test (for local debugging)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_queries = [
        "белая королева сериал 2013",
        "10 троллейбус ижевск",
        "смотреть фильмы 2025 года онлайн",
        "рик и морти 7 сезон",
        "как купить квартиру москва",
        "чебурашка смотреть онлайн",
        "слово пацана кровь на асфальте",
    ]
    df_test = pd.DataFrame({'QueryText': test_queries})

    model = PredictionModel()
    result = model.predict(df_test)
    print(result.to_string(index=False))
