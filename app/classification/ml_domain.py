"""Machine learning based domain classifier for determining search intent."""

import logging
from pathlib import Path
from typing import Any

import joblib

from app.preprocessing.normalizer import build_normalized_query

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("outputs/typequery_classifier.pkl")

class MachineLearningDomainClassifier:
    """Classifies search queries as video or non-video using a trained ML model.
    
    This classifier loads a scikit-learn pipeline (e.g. TF-IDF + LogisticRegression)
    and predicts the probability of the query being professional video content.
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        """Initialize the classifier and attempt to load the pre-trained model."""
        self.model_path = Path(model_path)
        self.pipeline: Any = None
        
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded ML model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model from {self.model_path}: {e}")
        else:
            logger.warning(
                f"ML model not found at {self.model_path}. "
                "Classifier will return base probabilities until trained."
            )

    def predict_probability(self, query: str) -> float | None:
        """Predict the probability that the query belongs to the video domain.
        
        Args:
            query: The raw search query string.
            
        Returns:
            A float probabilty between 0.0 and 1.0 representing the confidence
            that the query is video content. Returns None if the model is not loaded.
        """
        if self.pipeline is None:
            return None
            
        # We ensure the pipeline receives normalized text identically
        # to how it was trained.
        normalized = build_normalized_query(query).normalized_text
        if not normalized.strip():
            return 0.0
            
        try:
            # predict_proba returns an array of shape (n_samples, n_classes).
            # The classes are typically [False, True] (i.e. Non-Video, Video).
            probabilities = self.pipeline.predict_proba([normalized])[0]
            # Assumes the classes are sorted correctly or we extract the 
            # probability of the positive class (often index 1 or using classes_)
            # We enforce classes pattern during training.
            if hasattr(self.pipeline, "classes_"):
                positive_idx = list(self.pipeline.classes_).index(True)
                return float(probabilities[positive_idx])
            return float(probabilities[1])
        except Exception as e:
            logger.error(f"Prediction failed for query '{query}': {e}")
            return None
