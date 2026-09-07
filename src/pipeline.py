"""End-to-end enrichment pipeline for batch and interactive use."""

from __future__ import annotations

import pandas as pd

from .entities import EntityKeywordExtractor
from .sentiment import TransformerSentimentClassifier


class FeedbackInsightsPipeline:
    def __init__(self, sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment = TransformerSentimentClassifier(sentiment_model)
        self.extractor = EntityKeywordExtractor()

    def analyze_texts(self, texts: list[str]) -> pd.DataFrame:
        sentiment = self.sentiment.predict(texts)
        rows = []
        for text, result in zip(texts, sentiment):
            entities = self.extractor.extract_entities(text)
            rows.append({
                "text": text,
                "sentiment": result.label,
                "sentiment_confidence": round(result.confidence, 4),
                "sentiment_probabilities": result.probabilities,
                "entities": [entity.to_dict() for entity in entities],
                "entity_text": ", ".join(entity.text for entity in entities),
                "entity_labels": ", ".join(entity.label for entity in entities),
                "keywords": self.extractor.extract_keywords(text),
            })
        return pd.DataFrame(rows)

