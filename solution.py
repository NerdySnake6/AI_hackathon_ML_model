import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class PredictionModel:
    batch_size: int = 100

    def __init__(self) -> None:
        """Initialize the model and load necessary artifacts."""
        base_dir = Path(__file__).parent.resolve()
        
        typequery_path = base_dir / "outputs" / "typequery_classifier.pkl"
        contenttype_path = base_dir / "outputs" / "contenttype_classifier.pkl"
        titles_path = base_dir / "outputs" / "titles.json"

        # Load models
        try:
            self.typequery_model = joblib.load(typequery_path)
            self.contenttype_model = joblib.load(contenttype_path)
            
            with open(titles_path, 'r', encoding='utf-8') as f:
                self.known_titles = json.load(f)
                
            logger.info("Successfully loaded models and titles.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.typequery_model = None
            self.contenttype_model = None
            self.known_titles = []

    def extract_title(self, query: str) -> str:
        """Find the most likely title from the query using fuzzy matching."""
        if not query or not self.known_titles:
            return ""
            
        # Optimization: Exact match check
        query_lower = query.lower()
        for title in self.known_titles:
            if title.lower() in query_lower:
                return title
                
        # If no substring match, use rapidfuzz
        # We use fuzz.partial_ratio since the query contains more words than just the title
        match = process.extractOne(
            query, 
            self.known_titles, 
            scorer=fuzz.partial_ratio,
            score_cutoff=85
        )
        if match:
            return match[0]
        return ""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict TypeQuery, ContentType, and Title for a batch of queries."""
        if 'QueryText' not in df.columns:
            raise ValueError("DataFrame must contain 'QueryText' column")

        results = pd.DataFrame()
        results['QueryText'] = df['QueryText']
        
        if self.typequery_model is None or self.contenttype_model is None:
            # Fallback if models failed to load
            results['TypeQuery'] = 0
            results['Title'] = ""
            results['ContentType'] = "прочее"
            return results

        queries = df['QueryText'].fillna("").astype(str)
        
        # 1. Predict TypeQuery
        type_query_preds = self.typequery_model.predict(queries)
        results['TypeQuery'] = type_query_preds

        # 2. Predict ContentType
        content_type_preds = self.contenttype_model.predict(queries)
        results['ContentType'] = content_type_preds
        
        # Override ContentType for non-video queries
        results.loc[results['TypeQuery'] == 0, 'ContentType'] = ""

        # 3. Extract Titles
        titles = []
        for q, tq in zip(queries, type_query_preds):
            if tq == 1:
                title = self.extract_title(q)
                titles.append(title)
            else:
                titles.append("")
        
        results['Title'] = titles

        return results[['QueryText', 'TypeQuery', 'Title', 'ContentType']]
