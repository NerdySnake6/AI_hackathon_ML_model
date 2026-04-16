"""Unit tests for the Machine Learning Domain Classifier."""

import unittest
from unittest.mock import Mock, patch

from app.classification.ml_domain import MachineLearningDomainClassifier


class TestMachineLearningDomainClassifier(unittest.TestCase):
    """Test suite for the ML domain classifier using standard unittest."""

    @patch("app.classification.ml_domain.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classifier_initialization_without_model(self, mock_exists, mock_load):
        """Test safe initialization when the .pkl model does not exist."""
        mock_exists.return_value = False
        
        classifier = MachineLearningDomainClassifier(model_path="fake_path.pkl")
        self.assertIsNone(classifier.pipeline)
        
        # Should safely return None probability if not trained
        prob = classifier.predict_probability("смотреть фильмы")
        self.assertIsNone(prob)
        mock_load.assert_not_called()

    @patch("app.classification.ml_domain.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classifier_predict_probability(self, mock_exists, mock_load):
        """Test prediction probability when a model is correctly loaded."""
        mock_exists.return_value = True
        
        # Mock the scikit-learn pipeline
        mock_pipeline = Mock()
        # Assume [False, True] probabilities order and we pass [0.2, 0.8]
        mock_pipeline.predict_proba.return_value = [[0.2, 0.8]]
        mock_pipeline.classes_ = [False, True]
        mock_load.return_value = mock_pipeline
        
        classifier = MachineLearningDomainClassifier(model_path="fake_path.pkl")
        self.assertIsNotNone(classifier.pipeline)
        
        prob = classifier.predict_probability("интерстеллар смотреть онлайн")
        
        self.assertEqual(prob, 0.8)
        mock_pipeline.predict_proba.assert_called_once()
        
    @patch("app.classification.ml_domain.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classifier_handles_empty_query(self, mock_exists, mock_load):
        """Test behaviour when query normalizes to an empty string."""
        mock_exists.return_value = True
        mock_pipeline = Mock()
        mock_load.return_value = mock_pipeline
        
        classifier = MachineLearningDomainClassifier(model_path="fake_path.pkl")
        
        # "    " normalizes to empty string. It should return 0.0 directly.
        prob = classifier.predict_probability("    ")
        
        self.assertEqual(prob, 0.0)
        mock_pipeline.predict_proba.assert_not_called()

