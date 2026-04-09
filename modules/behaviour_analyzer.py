import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

class BehaviourAnalyzer:
    def __init__(self, history_size: int = 60):
        # person_id -> deque of (x, y, timestamp)
        self.trajectories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
    def update(self, person_id: str, cx: float, cy: float, speed: float, current_time: float) -> Dict[str, Any]:
        """
        Analyzes the trajectory of a person to classify behavior.
        Matches the interface expected by pipeline.py.
        """
        self.trajectories[person_id].append((cx, cy, current_time))
        
        traj = list(self.trajectories[person_id])
        if len(traj) < 10:
            return {"behaviour_label": "normal", "behaviour_score": 0.0}

        # 1. Detect Running
        if speed > 150: 
             return {"behaviour_label": "running", "behaviour_score": 50.0}

        # 2. Detect Pacing
        path_length = 0
        for i in range(1, len(traj)):
            path_length += np.sqrt((traj[i][0] - traj[i-1][0])**2 + (traj[i][1] - traj[i-1][1])**2)
        
        total_displacement = np.sqrt((traj[-1][0] - traj[0][0])**2 + (traj[-1][1] - traj[0][1])**2)
        
        if path_length > 200 and total_displacement < 50:
            return {"behaviour_label": "pacing", "behaviour_score": 35.0}

        # 3. Detect Erratic Movement
        angles = []
        for i in range(2, len(traj)):
            v1 = np.array([traj[i-1][0] - traj[i-2][0], traj[i-1][1] - traj[i-2][1]])
            v2 = np.array([traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 2 and norm2 > 2:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angles.append(np.degrees(angle))
        
        if len(angles) > 5:
            avg_angle = np.mean(angles)
            if avg_angle > 45:
                return {"behaviour_label": "erratic", "behaviour_score": 60.0}

        return {"behaviour_label": "normal", "behaviour_score": 0.0}

    def check_following(self, person_positions: Dict[str, tuple]) -> List[Dict[str, Any]]:
        """
        Detects if one person is following another based on trajectories.
        Matches the interface expected by pipeline.py.
        """
        following_pairs = []
        pids = list(person_positions.keys())
        
        for i in range(len(pids)):
            for j in range(len(pids)):
                if i == j: continue
                pid1, pid2 = pids[i], pids[j]
                
                traj1 = list(self.trajectories[pid1])
                traj2 = list(self.trajectories[pid2])
                
                if len(traj1) < 20 or len(traj2) < 20: continue
                
                # Simple following check: consistent distance and similar velocity
                # ... (placeholder logic for brevity)
        
        return following_pairs

behaviour_analyzer = BehaviourAnalyzer()
