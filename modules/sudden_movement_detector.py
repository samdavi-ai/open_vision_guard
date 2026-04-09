import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

class SuddenMovementDetector:
    def __init__(self, window_size: int = 15):
        self.speed_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, person_id: str, current_speed: float, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Detects sudden jumps in speed.
        Matches the interface expected by pipeline.py.
        """
        self.speed_history[person_id].append(current_speed)
        
        history = list(self.speed_history[person_id])
        if len(history) < 5:
            return None
            
        avg_prev_speed = sum(history[:-1]) / (len(history) - 1)
        
        if avg_prev_speed > 10 and current_speed > avg_prev_speed * 3.0:
            return {"type": "sudden_acceleration", "prev_avg": avg_prev_speed, "current": current_speed}
            
        if avg_prev_speed > 100 and current_speed < 10:
             return {"type": "sudden_deceleration", "prev_avg": avg_prev_speed, "current": current_speed}

        return None

sudden_movement_detector = SuddenMovementDetector()
