"""
Risk Engine Module
Aggregates all intelligence signals into a dynamic composite risk score per person.
"""
from typing import Dict, Any, List, Optional


class RiskEngine:
    """Computes a composite risk score from multiple signal sources."""

    # Weighted risk contributions
    WEIGHTS = {
        "weapon_proximity": 40,
        "camera_avoidance": 25,
        "erratic_behaviour": 20,
        "loitering": 15,
        "unknown_face": 10,
        "high_frequency": 5,
        "following_someone": 15,
        "sudden_movement": 20,
        "luggage_abandoned": 30,
        "luggage_theft": 35,
        "prolonged_stillness": 10,
        "pacing": 10,
        "circle_walking": 12,
        "running": 8,
    }

    # Risk level thresholds
    LEVELS = [
        (0, 20, "low"),
        (21, 45, "medium"),
        (46, 70, "high"),
        (71, 100, "critical"),
    ]

    def __init__(self):
        # person_id → {risk_score, risk_level, risk_factors[], last_update}
        self._scores: Dict[str, Dict[str, Any]] = {}

    def compute_risk(self, person_id: str, signals: Dict[str, bool],
                     behaviour_score: float = 0.0,
                     avoidance_score: float = 0.0) -> Dict[str, Any]:
        """
        Compute composite risk score from all active signals.

        Args:
            person_id: The person's global ID.
            signals: Dict of signal_name → bool (True if active).
            behaviour_score: Raw behaviour abnormality score (0-100).
            avoidance_score: Camera avoidance score (0-100).

        Returns:
            {"risk_score": float, "risk_level": str, "risk_factors": list}
        """
        total = 0.0
        factors = []

        for signal_name, is_active in signals.items():
            if is_active and signal_name in self.WEIGHTS:
                weight = self.WEIGHTS[signal_name]
                total += weight
                factors.append(signal_name)

        # Blend in continuous scores (behaviour + avoidance contribute proportionally)
        if behaviour_score > 20:
            behaviour_contrib = (behaviour_score / 100.0) * 15.0
            total += behaviour_contrib
            if "erratic_behaviour" not in factors:
                factors.append("abnormal_behaviour")

        if avoidance_score > 20:
            avoidance_contrib = (avoidance_score / 100.0) * 20.0
            total += avoidance_contrib
            if "camera_avoidance" not in factors:
                factors.append("camera_avoidance_score")

        # Clamp to 0-100
        risk_score = min(max(total, 0.0), 100.0)
        risk_level = self._get_level(risk_score)

        result = {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "risk_factors": factors,
        }

        self._scores[person_id] = result
        return result

    def get_risk(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self._scores.get(person_id)

    def should_alert(self, person_id: str, threshold: float = 70.0) -> bool:
        """Returns True if this person's risk crosses the alert threshold."""
        data = self._scores.get(person_id)
        if data:
            return data["risk_score"] >= threshold
        return False

    def _get_level(self, score: float) -> str:
        for lo, hi, level in self.LEVELS:
            if lo <= score <= hi:
                return level
        return "critical" if score > 100 else "low"


# Singleton
risk_engine = RiskEngine()
