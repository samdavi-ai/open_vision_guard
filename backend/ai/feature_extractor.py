import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DeepFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load a pre-trained MobileNetV2 (efficient for Re-ID features)
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classification head to get raw features (1280-dim)
        self.feature_extractor = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, cv2_image):
        """
        Extracts a 1280-dimensional feature vector from a CV2 image crop.
        """
        # Convert BGR (CV2) to RGB (PIL)
        image_rgb = cv2_image[:, :, ::-1]
        pil_img = Image.fromarray(image_rgb)
        
        # Preprocess and move to device
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.feature_extractor(img_tensor)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Normalize the feature vector
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / norm
        
        return features.cpu().numpy().flatten()
