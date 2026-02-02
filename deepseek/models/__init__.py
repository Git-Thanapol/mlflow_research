# models/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from einops import rearrange

# Base Model
class BaseModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
    def forward(self, x, metadata=None):
        raise NotImplementedError

# ==================== Simple CNN Blocks ====================
class SimpleConvBlock(nn.Module):
    """Basic convolution block for audio spectrograms"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=kernel_size, stride=stride, 
                            padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ==================== 1. Baseline CNN (Custom ResNet-style) ====================
class ResidualBlock(nn.Module):
    """Custom residual block without pretrained weights"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class BaselineCNN(BaseModel):
    """Custom ResNet-style CNN without pretrained weights"""
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(512, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, metadata=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# ==================== 2. CNN + MLP ====================
class SimpleCNNExtractor(nn.Module):
    """Simple CNN feature extractor for audio spectrograms"""
    def __init__(self, config):
        super().__init__()
        
        self.feature_dim = 256
        
        # Simple convolutional layers
        self.features = nn.Sequential(
            SimpleConvBlock(3, 32),  # 32 channels
            nn.MaxPool2d(2, 2),
            
            SimpleConvBlock(32, 64),  # 64 channels
            nn.MaxPool2d(2, 2),
            
            SimpleConvBlock(64, 128),  # 128 channels
            nn.MaxPool2d(2, 2),
            
            SimpleConvBlock(128, 256),  # 256 channels
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        return features


class AudioMetadataEncoder(nn.Module):
    """MLP for encoding audio metadata"""
    def __init__(self, config):
        super().__init__()
        
        metadata_dim = getattr(config, 'metadata_dim', 10)
        
        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.output_dim = 32
        
    def forward(self, metadata):
        return self.mlp(metadata)


class CNNMLP(BaseModel):
    """CNN + MLP model with simple CNN backbone (not ResNet)"""
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Visual feature extractor
        self.visual_extractor = SimpleCNNExtractor(config)
        
        # Metadata encoder
        self.metadata_encoder = AudioMetadataEncoder(config)
        
        # Combined classifier
        combined_dim = self.visual_extractor.feature_dim + self.metadata_encoder.output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
        
    def forward(self, x, metadata=None):
        # Extract visual features
        visual_features = self.visual_extractor(x)
        
        # Process metadata if available
        if metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            combined = torch.cat([visual_features, metadata_features], dim=1)
        else:
            # Use zeros if no metadata
            batch_size = x.size(0)
            device = x.device
            metadata_features = torch.zeros(batch_size, 32, device=device)
            combined = torch.cat([visual_features, metadata_features], dim=1)
            
        return self.classifier(combined)


# ==================== 3. CNN + Attention (CBAM) ====================
class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return attention * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CNNAttention(BaseModel):
    """Custom CNN with CBAM attention (no pretrained weights)"""
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Custom blocks with attention
        self.block1 = self._make_block(64, 64, stride=1, with_attention=True)
        self.block2 = self._make_block(64, 128, stride=2, with_attention=True)
        self.block3 = self._make_block(128, 256, stride=2, with_attention=True)
        self.block4 = self._make_block(256, 512, stride=2, with_attention=True)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(512, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_block(self, in_channels, out_channels, stride, with_attention):
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Add CBAM after second conv
        if with_attention:
            layers.append(CBAM(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, metadata=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# ==================== 4. Audio Spectrogram Transformer (AST) ====================
class PatchEmbedding(nn.Module):
    """Split spectrogram into patches and embed them"""
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Calculate number of patches
        f, t = config.ast_input_size
        pf, pt = config.ast_patch_size
        self.num_patches = (f // pf) * (t // pt)
        
        self.projection = nn.Conv2d(
            1, config.ast_dim,
            kernel_size=config.ast_patch_size,
            stride=config.ast_patch_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.ast_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.ast_dim)
        )
        
    def forward(self, x):
        # x: (B, 1, Freq, Time)
        B = x.shape[0]
        
        # Project patches
        x = self.projection(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x += self.position_embedding
        
        return x


class AST(BaseModel):
    """Audio Spectrogram Transformer without pretrained weights"""
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.ast_dim,
            nhead=config.ast_heads,
            dim_feedforward=config.ast_dim * 4,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.ast_depth
        )
        
        # Classifier head
        self.norm = nn.LayerNorm(config.ast_dim)
        self.head = nn.Sequential(
            nn.Linear(config.ast_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embedding_dim, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize patch embedding
        nn.init.kaiming_normal_(self.patch_embed.projection.weight, 
                               mode='fan_out', nonlinearity='relu')
        
        # Initialize transformer
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize classifier
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, metadata=None):
        # Convert RGB to single channel for AST
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # Convert to grayscale
        
        # Resize if needed
        if x.shape[2:] != self.config.ast_input_size:
            x = F.interpolate(x, size=self.config.ast_input_size, mode='bilinear')
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification token
        x = x[:, 0]  # CLS token
        x = self.norm(x)
        
        return self.head(x)


# ==================== Model Factory ====================
def get_model(config: Any) -> BaseModel:
    """Factory function to get model based on config"""
    model_registry = {
        "baseline_cnn": BaselineCNN,
        "cnn_mlp": CNNMLP,
        "cnn_attention": CNNAttention,
        "ast": AST
    }
    
    if config.model_type not in model_registry:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model_registry[config.model_type](config)