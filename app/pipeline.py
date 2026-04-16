"""End-to-end query labeling pipeline."""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from app.catalog.repository import InMemoryCatalogRepository
from app.classification.domain import (
    RuleBasedDomainClassifier,
    RuleBasedGenericDetector,
)
from app.classification.ml_domain import MachineLearningDomainClassifier
from app.db.repositories import CatalogDatabaseRepository
from app.db.session import DATABASE_URL, build_engine
from app.policy.decision import ConfidencePolicy
from app.preprocessing.normalizer import build_normalized_query
from app.retrieval.hybrid import HybridCandidateRetriever
from app.ranking.ranker import RuleBasedCandidateRanker
from app.schemas import (
    ContentType,
    Decision,
    DomainLabel,
    LabelResult,
    QueryInput,
)
from app.settings import DEFAULT_CATALOG_PATH, MAX_CANDIDATES


class QueryLabelingPipeline:
    """Coordinate preprocessing, catalog search, classification, and policy."""

    def __init__(
        self,
        catalog: InMemoryCatalogRepository,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        """Initialize the query labeling pipeline."""
        self.catalog = catalog
        self.max_candidates = max_candidates
        self.retriever = HybridCandidateRetriever.from_catalog(catalog)
        self.ranker = RuleBasedCandidateRanker()
        self.ml_classifier = MachineLearningDomainClassifier()
        self.domain_classifier = RuleBasedDomainClassifier(ml_classifier=self.ml_classifier)
        self.generic_detector = RuleBasedGenericDetector()
        self.policy = ConfidencePolicy()

    @classmethod
    def from_csv(
        cls,
        catalog_path: str = DEFAULT_CATALOG_PATH,
        max_candidates: int = MAX_CANDIDATES,
    ) -> "QueryLabelingPipeline":
        """Create a pipeline from a CSV catalog."""
        return cls(
            catalog=InMemoryCatalogRepository.from_csv(catalog_path),
            max_candidates=max_candidates,
        )

    @classmethod
    def from_db(
        cls,
        database_url: str = DATABASE_URL,
        max_candidates: int = MAX_CANDIDATES,
    ) -> "QueryLabelingPipeline":
        """Create a pipeline from catalog titles stored in the configured DB."""
        engine = build_engine(database_url)
        session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        try:
            with session_factory() as session:
                repository = CatalogDatabaseRepository(session)
                titles = repository.list_titles()
        finally:
            engine.dispose()

        return cls(
            catalog=InMemoryCatalogRepository(titles),
            max_candidates=max_candidates,
        )

    def label(self, query: QueryInput | str) -> LabelResult:
        """Label one query and return the final structured result."""
        query_input = query if isinstance(query, QueryInput) else QueryInput(query_text=query)
        normalized = build_normalized_query(query_input.query_text)
        lexical_candidates = self.catalog.search(
            normalized,
            limit=self.max_candidates,
        )
        found_candidates = self.retriever.search(
            normalized,
            lexical_candidates=lexical_candidates,
            limit=self.max_candidates,
        )
        ranked_candidates = self.ranker.rank(normalized, found_candidates)
        domain = self.domain_classifier.predict(normalized, ranked_candidates)
        generic = self.generic_detector.predict(normalized, ranked_candidates)
        decision = self.policy.decide(domain, generic, ranked_candidates)

        top_candidate = ranked_candidates[0] if ranked_candidates else None
        is_prof_video = DomainLabel(domain.label) == DomainLabel.PROF_VIDEO
        content_type: ContentType | None = None
        title: str | None = None
        title_id: str | None = None
        confidence = domain.confidence

        if decision == Decision.NON_VIDEO:
            is_prof_video = False
        elif decision == Decision.GENERIC_VIDEO:
            is_prof_video = True
            content_type = generic.content_type or ContentType.GENERIC
            confidence = generic.confidence
        elif top_candidate is not None:
            is_prof_video = True
            content_type = top_candidate.content_type
            title = top_candidate.canonical_title
            title_id = top_candidate.title_id
            confidence = top_candidate.rank_score

        reasons = [
            *domain.reasons,
            *generic.reasons,
            *(top_candidate.reasons if top_candidate else []),
        ]

        return LabelResult(
            query_id=query_input.query_id,
            query=query_input.query_text,
            normalized_query=normalized.normalized_text,
            is_prof_video=is_prof_video,
            domain_label=DomainLabel(domain.label),
            content_type=content_type,
            title=title,
            title_id=title_id,
            confidence=confidence,
            decision=decision,
            candidates=ranked_candidates,
            reasons=list(dict.fromkeys(reasons)),
        )
