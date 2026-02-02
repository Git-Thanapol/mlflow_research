# models/cnn_mlp_variants.py
class CNNMLPResidualFusion(nn.Module):
    """CNN+MLP with residual connection for metadata"""
    def __init__(self, config):
        super().__init__()
        
        # CNN backbone (can be ResNet or simpler)
        self.cnn = SimpleCNNExtractor(config)
        
        # Metadata MLP
        self.metadata_mlp = AudioMetadataEncoder(config)
        
        # Separate classifiers for each modality
        self.visual_classifier = nn.Linear(256, config.num_classes)
        self.metadata_classifier = nn.Linear(32, config.num_classes)
        self.joint_classifier = nn.Linear(config.num_classes * 2, config.num_classes)
    
    def forward(self, image, metadata=None):
        # Extract features
        visual_features = self.cnn(image)
        visual_logits = self.visual_classifier(visual_features)
        
        if metadata is not None:
            metadata_features = self.metadata_mlp(metadata)
            metadata_logits = self.metadata_classifier(metadata_features)
            
            # Combine logits (not features)
            combined_logits = torch.cat([visual_logits, metadata_logits], dim=1)
            output = self.joint_classifier(combined_logits)
        else:
            output = visual_logits
        
        return output


class CNNMLPCrossAttention(nn.Module):
    """CNN+MLP with cross-attention between modalities"""
    def __init__(self, config):
        super().__init__()
        
        self.visual_extractor = SimpleCNNExtractor(config)
        self.metadata_encoder = AudioMetadataEncoder(config)
        
        # Cross-attention layers
        self.visual_to_metadata = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True
        )
        self.metadata_to_visual = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Linear(256 + 32, config.num_classes)
    
    def forward(self, image, metadata=None):
        visual_features = self.visual_extractor(image)
        
        if metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            
            # Reshape for attention (add sequence dimension)
            visual_seq = visual_features.unsqueeze(1)  # [B, 1, 256]
            metadata_seq = metadata_features.unsqueeze(1)  # [B, 1, 32]
            
            # Cross-attention
            visual_attended, _ = self.visual_to_metadata(
                visual_seq, metadata_seq, metadata_seq
            )
            metadata_attended, _ = self.metadata_to_visual(
                metadata_seq, visual_seq, visual_seq
            )
            
            # Flatten
            visual_attended = visual_attended.squeeze(1)
            metadata_attended = metadata_attended.squeeze(1)
            
            combined = torch.cat([visual_attended, metadata_attended], dim=1)
        else:
            combined = visual_features
        
        return self.classifier(combined)