import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

class LuggageTracker:
    def __init__(self):
        # luggage_id -> {owner_id, type, last_pos, status, last_seen_time, abandoned_time}
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.abandon_threshold = 10 
        self.link_distance = 150     

    def update(self, object_detections: List[Dict[str, Any]], person_positions: Dict[str, tuple], current_time: float) -> Dict[str, Any]:
        """
        Updates luggage status and ownership.
        Matches the interface expected by pipeline.py.
        """
        events = []
        
        for obj in object_detections:
            if obj["type"] not in ["backpack", "handbag", "suitcase"]:
                continue
                
            obj_cx, obj_cy = obj["center"]
            
            # Find nearest person
            min_dist = float('inf')
            nearest_pid = None
            
            for pid, p_pos in person_positions.items():
                p_cx, p_cy = p_pos
                dist = ((obj_cx - p_cx)**2 + (obj_cy - p_cy)**2)**0.5
                
                if dist < min_dist and dist < self.link_distance:
                    min_dist = dist
                    nearest_pid = pid
            
            obj_id = f"Luggage_{int(obj_cx)}_{int(obj_cy)}"
            
            if nearest_pid:
                if obj_id not in self.registry:
                    self.registry[obj_id] = {
                        "owner_id": nearest_pid,
                        "type": obj["type"],
                        "last_pos": (obj_cx, obj_cy),
                        "status": "carried",
                        "last_seen_time": current_time,
                        "abandoned_time": None
                    }
                else:
                    linfo = self.registry[obj_id]
                    if linfo["owner_id"] != nearest_pid:
                         events.append({"type": "luggage_transferred", "owner_id": linfo["owner_id"], "new_holder": nearest_pid, "luggage_type": obj["type"]})
                         linfo["owner_id"] = nearest_pid
                    
                    linfo.update({
                        "last_pos": (obj_cx, obj_cy),
                        "status": "carried",
                        "last_seen_time": current_time,
                        "abandoned_time": None
                    })
            else:
                # Potential abandonment check
                for lid, linfo in self.registry.items():
                    dist = ((obj_cx - linfo["last_pos"][0])**2 + (obj_cy - linfo["last_pos"][1])**2)**0.5
                    if dist < 50:
                        linfo["last_seen_time"] = current_time
                        if linfo["status"] == "carried":
                            linfo["status"] = "abandoned"
                            linfo["abandoned_time"] = current_time
                            events.append({"type": "luggage_abandoned", "owner_id": linfo["owner_id"], "luggage_type": linfo["type"]})
                        break

        return {"events": events}

    def get_person_luggage(self, person_id: str) -> Dict[str, Any]:
        """
        Returns all luggage linked to a person.
        """
        res = {}
        for lid, linfo in self.registry.items():
            if linfo["owner_id"] == person_id:
                res[lid] = {
                    "type": linfo["type"],
                    "status": linfo["status"]
                }
        return res

luggage_tracker = LuggageTracker()
