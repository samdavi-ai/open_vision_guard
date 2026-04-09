from typing import List, Dict, Any, Optional

class RiskEngine:
    def __init__(self):
        # Weights for various risk factors
        self.weights = {
            "weapon_proximity": 50,
            "loitering": 15,
            "unknown_face": 10,
            "high_frequency": 5,
            "following_someone": 15,
            "prolonged_stillness": 10,
            "pacing": 10,
            "circle_walking": 20,
            "running": 20,
            "camera_avoidance": 25
        }
        self.risk_cache: Dict[str, float] = {}

    def compute_risk(self, person_id: str, signals: Dict[str, bool], behaviour_score: float = 0.0, avoidance_score: float = 0.0) -> Dict[str, Any]:
        """
        Computes a composite risk score (0-100) and identifies risk factors.
        Matches the interface expected by pipeline.py.
        """
        score = 0.0
        factors = []

        for signal, active in signals.items():
            if active:
                weight = self.weights.get(signal, 10)
                score += weight
                factors.append(signal.replace("_", " ").title())

        # Include continuous scores
        if behaviour_score > 30:
             score += behaviour_score * 0.5
        
        if avoidance_score > 30:
             score += avoidance_score * 0.5

        # Caps at 100
        score = min(score, 100.0)
        self.risk_cache[person_id] = score

        # Map score to risk level
        level = "low"
        if score > 70:
            level = "critical"
        elif score > 45:
            level = "high"
        elif score > 20:
            level = "medium"

        return {
            "risk_score": float(score),
            "risk_level": level,
            "risk_factors": factors
        }

    def should_alert(self, person_id: str, threshold: float = 70.0) -> bool:
        return self.risk_cache.get(person_id, 0.0) >= threshold

risk_engine = RiskEngine()
