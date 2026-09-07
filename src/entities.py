"""Named entities and lightweight, explainable topic extraction."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

import spacy


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


class EntityKeywordExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
            self.using_spacy = True
        except (OSError, ImportError):
            self.nlp = spacy.blank("en")
            self.using_spacy = False

    def extract_entities(self, text: str) -> list[Entity]:
        if self.using_spacy:
            doc = self.nlp(text)
            return [Entity(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        entities: list[Entity] = []
        patterns = {
            "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
            "ORDER_ID": r"\b(?:order|ticket|case)[\s#-]*\d+\b",
            "PRODUCT": r"\b(?:AirMax|dashboard|headphones|subscription|mobile app)\b",
            "LOCATION": r"\b(?:Toronto|Calgary|Vancouver|Montreal)\b",
        }
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                entities.append(Entity(match.group(), label, match.start(), match.end()))
        return sorted(entities, key=lambda item: item.start)

    def extract_keywords(self, text: str, top_k: int = 8) -> list[str]:
        doc = self.nlp(text)
        if self.using_spacy:
            candidates = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]
        else:
            stop = {
                "the", "and", "for", "with", "this", "that", "have", "has", "but", "not", "was", "from",
                "when", "while", "then", "than", "into", "over", "under", "after", "before", "just",
                "try", "tried", "trying", "get", "got", "make", "made", "use", "used", "using", "could",
                "would", "should", "will", "can", "please", "very", "also", "still", "one", "two", "my", "our",
            }
            candidates = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text) if w.lower() not in stop]
        counts = Counter(candidates)
        return [term for term, _ in counts.most_common(top_k)]

