"""
Enhanced Behavior Analysis Module
==================================
Comprehensive behavior detection for surveillance:
- Running, Walking, Loitering (standing still for extended time)
- Pacing (repetitive back-and-forth motion)
- Sudden abnormal movement (jerky, erratic changes)
- Temporal state tracking per person

Author: OpenVisionGuard ML Team
Date: April 2026
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TemporalState:
    """Track temporal state for a single person."""
    person_id: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    time_in_zone: float = 0.0
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=100))
    loitering_start: Optional[float] = None
    loitering_threshold: float = 30.0  # seconds
    stationary_threshold: float = 10.0  # pixels per frame
    
    def update_seen_time(self, current_time: float):
        """Update tracking timestamps."""
        if self.first_seen == 0.0:
            self.first_seen = current_time
        self.last_seen = current_time
        self.time_in_zone = current_time - self.first_seen


class BehaviourAnalyzer:
    """
    Advanced behavior analyzer with temporal state tracking and multiple detection methods.
    """
    
    def __init__(self, history_size: int = 120, fps: int = 30):
        """
        Args:
            history_size: Number of frames to track (history_size/fps = seconds)
            fps: Video frame rate (default 30)
        """
        # Trajectory storage: person_id -> deque of (x, y, timestamp, speed)
        self.trajectories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Temporal state tracking
        self.temporal_states: Dict[str, TemporalState] = {}
        
        # Configuration
        self.fps = fps
        self.history_size = history_size
        
        # Thresholds (configurable)
        self.running_speed_threshold = 150  # pixels/frame
        self.walking_speed_threshold = 30   # pixels/frame
        self.pacing_path_ratio = 2.0  # path_length / displacement
        self.loitering_time_threshold = 30  # seconds
        self.loitering_radius = 50  # pixels
        self.sudden_movement_angle_threshold = 60  # degrees
        self.erratic_avg_angle_threshold = 45  # degrees
        
    def update(self, person_id: str, cx: float, cy: float, speed: float, current_time: float) -> Dict[str, Any]:
        """
        Update behavior analysis for a person.
        
        Args:
            person_id: Unique person identifier
            cx, cy: Center coordinates
            speed: Current speed estimate
            current_time: Timestamp (seconds)
            
        Returns:
            Dictionary with behavior_label and behavior_score
        """
        # Update trajectory
        self.trajectories[person_id].append((cx, cy, current_time, speed))
        
        # Update or create temporal state
        if person_id not in self.temporal_states:
            self.temporal_states[person_id] = TemporalState(person_id=person_id)
        
        state = self.temporal_states[person_id]
        state.update_seen_time(current_time)
        
        # Get trajectory
        traj = list(self.trajectories[person_id])
        
        # Not enough data yet
        if len(traj) < 5:
            return {"behaviour_label": "unknown", "behaviour_score": 0.0}
        
        # Extract positions and speeds
        positions = [(t[0], t[1]) for t in traj]
        speeds = [t[3] for t in traj]
        
        # ====== BEHAVIOR DETECTION HIERARCHY ======
        
        # 1. RUNNING - Highest speed threshold
        if speed > self.running_speed_threshold:
            state.behavior_history.append("running")
            return {
                "behaviour_label": "running",
                "behaviour_score": 70.0,
                "time_in_zone": state.time_in_zone
            }
        
        # 2. SUDDEN ABNORMAL MOVEMENT - Sudden speed changes or direction changes
        sudden_behavior = self._detect_sudden_movement(positions, speeds, traj)
        if sudden_behavior and sudden_behavior['score'] > 0:
            state.behavior_history.append("sudden_movement")
            return {
                "behaviour_label": "sudden_movement",
                "behaviour_score": sudden_behavior['score'],
                "time_in_zone": state.time_in_zone
            }
        
        # 3. LOITERING - Standing still for extended time (CRITICAL)
        loitering_info = self._detect_loitering(positions, current_time, state)
        if loitering_info['is_loitering']:
            state.behavior_history.append("loitering")
            return {
                "behaviour_label": "loitering",
                "behaviour_score": loitering_info['score'],
                "time_in_zone": state.time_in_zone,
                "loitering_duration": loitering_info['duration'],
                "loitering_location": (loitering_info['center_x'], loitering_info['center_y'])
            }
        
        # 4. PACING - Repetitive back-and-forth motion
        pacing_info = self._detect_pacing(positions)
        if pacing_info['is_pacing']:
            state.behavior_history.append("pacing")
            return {
                "behaviour_label": "pacing",
                "behaviour_score": pacing_info['score'],
                "time_in_zone": state.time_in_zone
            }
        
        # 5. ERRATIC MOVEMENT - Inconsistent direction changes
        erratic_info = self._detect_erratic_movement(positions)
        if erratic_info['is_erratic']:
            state.behavior_history.append("erratic")
            return {
                "behaviour_label": "erratic",
                "behaviour_score": erratic_info['score'],
                "time_in_zone": state.time_in_zone
            }
        
        # 6. NORMAL behavior
        state.behavior_history.append("normal")
        return {
            "behaviour_label": "normal",
            "behaviour_score": 0.0,
            "time_in_zone": state.time_in_zone
        }
    
    # ========== BEHAVIOR DETECTION METHODS ==========
    
    def _detect_loitering(self, positions: List[tuple], current_time: float, state: TemporalState) -> Dict[str, Any]:
        """
        Detect loitering (standing in one location for extended time).
        Critical for surveillance: identifies suspicious standing behavior.
        """
        if len(positions) < 10:
            return {'is_loitering': False, 'score': 0.0, 'duration': 0.0, 'center_x': 0, 'center_y': 0}
        
        # Calculate centroid of recent positions
        recent_positions = positions[-30:]  # Last ~1 second at 30fps
        center_x = np.mean([p[0] for p in recent_positions])
        center_y = np.mean([p[1] for p in recent_positions])
        
        # Calculate radius (spread of positions)
        distances = [np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in recent_positions]
        avg_radius = np.mean(distances)
        
        # Loitering if person stays in small radius
        is_stationary = avg_radius < self.loitering_radius
        
        if is_stationary:
            # Start or continue loitering timer
            if state.loitering_start is None:
                state.loitering_start = current_time
            
            loitering_duration = current_time - state.loitering_start
            
            if loitering_duration >= self.loitering_time_threshold:
                # High risk: loitering detected
                score = min(100.0, 30.0 + loitering_duration * 2)  # Increase with duration
                return {
                    'is_loitering': True,
                    'score': score,
                    'duration': loitering_duration,
                    'center_x': float(center_x),
                    'center_y': float(center_y)
                }
            else:
                # Low-medium risk: on the edge of loitering
                return {
                    'is_loitering': True,
                    'score': (loitering_duration / self.loitering_time_threshold) * 40,
                    'duration': loitering_duration,
                    'center_x': float(center_x),
                    'center_y': float(center_y)
                }
        else:
            # Reset loitering timer if person moves
            state.loitering_start = None
            return {'is_loitering': False, 'score': 0.0, 'duration': 0.0, 'center_x': float(center_x), 'center_y': float(center_y)}
    
    def _detect_pacing(self, positions: List[tuple]) -> Dict[str, Any]:
        """
        Detect pacing (back-and-forth repetitive motion).
        Indicates anxiety, surveillance evasion, or preparation for crime.
        """
        if len(positions) < 20:
            return {'is_pacing': False, 'score': 0.0}
        
        # Calculate total path length
        path_length = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        # Calculate straight-line displacement
        total_displacement = np.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )
        
        # High path length relative to displacement = pacing
        if total_displacement > 0:
            path_ratio = path_length / total_displacement
        else:
            path_ratio = 0
        
        is_pacing = (path_length > 100 and path_ratio > self.pacing_path_ratio)
        
        if is_pacing:
            score = min(60.0, 20.0 + path_ratio * 10)
            return {'is_pacing': True, 'score': score}
        
        return {'is_pacing': False, 'score': 0.0}
    
    def _detect_erratic_movement(self, positions: List[tuple]) -> Dict[str, Any]:
        """
        Detect erratic movement (sudden direction changes, jerky motion).
        Indicates nervousness, evasion attempts, or threat behavior.
        """
        if len(positions) < 10:
            return {'is_erratic': False, 'score': 0.0}
        
        # Calculate direction vectors
        angles = []
        for i in range(2, len(positions)):
            v1 = np.array([positions[i-1][0] - positions[i-2][0],
                          positions[i-1][1] - positions[i-2][1]])
            v2 = np.array([positions[i][0] - positions[i-1][0],
                          positions[i][1] - positions[i-1][1]])
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 2 and norm2 > 2:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angles.append(np.degrees(angle))
        
        if len(angles) > 5:
            avg_angle = np.mean(angles)
            if avg_angle > self.erratic_avg_angle_threshold:
                return {'is_erratic': True, 'score': min(60.0, avg_angle / 2)}
        
        return {'is_erratic': False, 'score': 0.0}
    
    def _detect_sudden_movement(self, positions: List[tuple], speeds: List[float], traj: List[tuple]) -> Dict[str, Any]:
        """
        Detect sudden abnormal movement (spikes in speed or acceleration).
        """
        if len(speeds) < 3:
            return {'is_sudden': False, 'score': 0.0}
        
        # Calculate acceleration (speed changes)
        accelerations = []
        for i in range(1, len(speeds)):
            accel = abs(speeds[i] - speeds[i-1])
            accelerations.append(accel)
        
        avg_accel = np.mean(accelerations)
        max_accel = np.max(accelerations)
        
        # Sudden acceleration = high spike
        if max_accel > avg_accel * 3:  # 3x normal acceleration
            return {'is_sudden': True, 'score': min(50.0, max_accel / 100)}
        
        return {'is_sudden': False, 'score': 0.0}
    
    # ========== UTILITY METHODS ==========
    
    def check_following(self, person_positions: Dict[str, tuple]) -> List[Dict[str, Any]]:
        """
        Detects if one person is following another.
        """
        following_pairs = []
        pids = list(person_positions.keys())
        
        if len(pids) < 2:
            return following_pairs
        
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pid1, pid2 = pids[i], pids[j]
                
                traj1 = list(self.trajectories[pid1])
                traj2 = list(self.trajectories[pid2])
                
                if len(traj1) < 20 or len(traj2) < 20:
                    continue
                
                # Check distance consistency and velocity similarity
                recent1 = traj1[-20:]
                recent2 = traj2[-20:]
                
                distances = []
                for (x1, y1, _, _), (x2, y2, _, _) in zip(recent1, recent2):
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    distances.append(dist)
                
                # Following if consistent distance (not too close, not diverging)
                std_dist = np.std(distances)
                avg_dist = np.mean(distances)
                
                if avg_dist > 50 and std_dist < avg_dist * 0.3:
                    following_pairs.append({
                        'follower': pid1,
                        'leader': pid2,
                        'confidence': min(1.0, 1.0 - (std_dist / avg_dist))
                    })
        
        return following_pairs
    
    def get_person_behavior_summary(self, person_id: str) -> Dict[str, Any]:
        """
        Get summary of person's behavior over time.
        """
        if person_id not in self.temporal_states:
            return {}
        
        state = self.temporal_states[person_id]
        behavior_hist = list(state.behavior_history)
        
        if not behavior_hist:
            return {'primary_behavior': 'unknown', 'time_in_zone': state.time_in_zone}
        
        # Count behaviors
        behavior_counts = defaultdict(int)
        for b in behavior_hist:
            behavior_counts[b] += 1
        
        primary = max(behavior_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'person_id': person_id,
            'primary_behavior': primary,
            'behavior_history': behavior_hist[-20:],  # Last 20 frames worth
            'time_in_zone': state.time_in_zone,
            'first_seen': state.first_seen,
            'last_seen': state.last_seen
        }
    
    def cleanup_old_states(self, current_time: float, timeout_seconds: float = 300):
        """
        Remove temporal states for people no longer tracked.
        """
        to_delete = []
        for person_id, state in self.temporal_states.items():
            if current_time - state.last_seen > timeout_seconds:
                to_delete.append(person_id)
        
        for person_id in to_delete:
            del self.temporal_states[person_id]
            if person_id in self.trajectories:
                del self.trajectories[person_id]


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

behaviour_analyzer = BehaviourAnalyzer()
