# models/cnn_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleCNNExtractor(nn.Module):
    """
    Simple CNN feature extractor for audio spectrograms.
    More basic than ResNet, designed to work with metadata MLP.
    """
    def __init__(self, config):
        super().__init__()
        
        # Convolutional layers (simpler than ResNet)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate output dimension
        self.feature_dim = 256
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        features = torch.flatten(features, 1)
        return features


class AudioMetadataEncoder(nn.Module):
    """MLP for encoding audio metadata"""
    def __init__(self, config, metadata_dim: int = 10):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.metadata_dim = metadata_dim
        self.output_dim = 32
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.mlp(metadata)


class CNNMLP(nn.Module):
    """
    CNN + MLP model for audio spectrogram classification.
    Combines visual features from spectrograms with audio metadata.
    """
    def __init__(self, config):
        super().__init__()
        
        # Visual feature extractor (simpler CNN)
        self.visual_extractor = SimpleCNNExtractor(config)
        
        # Metadata encoder (MLP)
        self.metadata_encoder = AudioMetadataEncoder(config)
        
        # Feature fusion and classification
        combined_dim = self.visual_extractor.feature_dim + self.metadata_encoder.output_dim
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(256, config.num_classes)
        )
        
        # Alternative: Late fusion with attention
        if config.model_type == "cnn_mlp_attention":
            self.feature_attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ReLU(),
                nn.Linear(combined_dim // 2, 2),
                nn.Softmax(dim=1)
            )
    
    def forward(self, 
                image: torch.Tensor, 
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Extract visual features
        visual_features = self.visual_extractor(image)
        
        # Process metadata if available
        if metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            
            # Concatenate features
            combined_features = torch.cat([visual_features, metadata_features], dim=1)
        else:
            # Use zeros for metadata if not provided
            batch_size = image.size(0)
            metadata_features = torch.zeros(
                batch_size, 
                self.metadata_encoder.output_dim, 
                device=image.device
            )
            combined_features = torch.cat([visual_features, metadata_features], dim=1)
        
        # Classify
        return self.fusion_classifier(combined_features)