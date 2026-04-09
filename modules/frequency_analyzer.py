from typing import Dict, List, Any, Optional
from modules import database

class FrequencyAnalyzer:
    def __init__(self):
        # We can cache visit counts to reduce DB load
        self.cache: Dict[str, int] = {}

    def record_appearance(self, person_id: str, current_time: float):
        """
        In a real system, we might only log appearance once per session.
        For now, this is a placeholder for session-based appearance logging.
        """
        pass

    def get_frequency_data(self, person_id: str) -> Dict[str, Any]:
        """
        Returns frequency metrics for a person.
        Matches the interface expected by pipeline.py.
        """
        # Fetch from DB if not in cache (simplified)
        history = database.get_visit_history(person_id)
        visit_count = len(history)
        
        label = "new"
        if visit_count > 10: label = "regular"
        elif visit_count > 5: label = "frequent"
        elif visit_count > 2: label = "occasional"
        elif visit_count >= 1: label = "rare"

        return {
            "visit_count": visit_count,
            "frequency_label": label
        }

    def is_frequent(self, person_id: str) -> bool:
        data = self.get_frequency_data(person_id)
        return data["visit_count"] > 3

frequency_analyzer = FrequencyAnalyzer()
