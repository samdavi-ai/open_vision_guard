import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Any, Optional
import datetime
from config import config

class EmbeddingEngine:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained MobileNetV2
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add a projection head to get 512-dim embedding as per requirements
        # MobileNetV2 features output 1280 channels
        self.projection = nn.Linear(1280, 512)
        
        self.backbone.to(self.device).eval()
        self.projection.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # In-memory identity registry: global_id -> {embedding, metadata, history}
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.next_id_counter = 1

    @torch.no_grad()
    def generate_embedding(self, cv2_image: np.ndarray) -> np.ndarray:
        """Generates a 512-dim L2-normalized embedding from a crop."""
        image_rgb = cv2_image[:, :, ::-1] # BGR to RGB
        pil_img = Image.fromarray(image_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        features = self.backbone(img_tensor)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        embedding = self.projection(features)
        
        # L2 Normalization
        norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding = embedding / (norm + 1e-6)
        
        return embedding.cpu().numpy().flatten()

    def get_or_create_identity(self, crop_img: np.ndarray) -> str:
        """Matches a crop to an existing identity or creates a new one."""
        new_embedding = self.generate_embedding(crop_img)
        
        best_match_id = None
        best_sim = -1.0
        
        for gid, data in self.registry.items():
            sim = 1.0 - cosine(new_embedding, data['embedding'])
            if sim > best_sim:
                best_sim = sim
                best_match_id = gid
        
        if best_sim > config.similarity_threshold:
            # Update last seen and potentially refine embedding (moving average)
            self.registry[best_match_id]['metadata']['last_seen_time'] = datetime.datetime.now().isoformat()
            # Optional: Smooth embedding
            self.registry[best_match_id]['embedding'] = 0.9 * self.registry[best_match_id]['embedding'] + 0.1 * new_embedding
            # Re-normalize
            self.registry[best_match_id]['embedding'] /= (np.linalg.norm(self.registry[best_match_id]['embedding']) + 1e-6)
            return best_match_id
        
        # Create new identity
        new_id = f"Person_{self.next_id_counter:03d}"
        self.next_id_counter += 1
        
        self.registry[new_id] = {
            "global_id": new_id,
            "embedding": new_embedding,
            "metadata": {
                "face_name": None,
                "activity": "unknown",
                "risk_level": "low",
                "clothing_color": "unknown",
                "last_seen_camera": "unknown",
                "last_seen_time": datetime.datetime.now().isoformat()
            },
            "history": []
        }
        return new_id

    def update_identity_metadata(self, global_id: str, updates: Dict[str, Any]):
        """Updates metadata for a specific identity."""
        if global_id in self.registry:
            self.registry[global_id]['metadata'].update(updates)

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Searches for the most similar identities in the registry."""
        results = []
        for gid, data in self.registry.items():
            sim = 1.0 - cosine(query_embedding, data['embedding'])
            results.append((gid, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [gid for gid, sim in results[:top_k]]

    def get_all_identities(self) -> List[Dict[str, Any]]:
        return list(self.registry.values())

    def get_identity(self, global_id: str) -> Optional[Dict[str, Any]]:
        return self.registry.get(global_id)

# Singleton instance
embedding_engine = EmbeddingEngine()
