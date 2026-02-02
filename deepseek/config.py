# config.py
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

@dataclass
class Config:
    """Configuration for audio spectrogram classification framework"""
    # Data parameters

    data_dir: str = str(Path(__file__).resolve().parent.parent / "mel_spectrograms")
    image_size: Tuple[int, int] = (384, 384)
    num_classes: int = 10
    batch_size: int = 32
    num_workers: int = 4
    k_folds: int = 5
    seed: int = 42
    
    # Audio specific
    sample_rate: int = 44100
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    # Model parameters
    model_type: str = "baseline_cnn"  # baseline_cnn, cnn_mlp, cnn_attention, ast
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout_rate: float = 0.3
    embedding_dim: int = 512
    
    # AST specific
    ast_input_size: Tuple[int, int] = (128, 1000)  # (freq_bins, time_frames)
    ast_patch_size: Tuple[int, int] = (16, 16)
    ast_dim: int = 768
    ast_depth: int = 12
    ast_heads: int = 12
    
    # Training parameters
    epochs: int = 50
    learning_rate: float = 1e-3
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Augmentation
    freq_mask_param: int = 15
    time_mask_param: int = 15
    num_masks: int = 2
    
    # MLflow
    experiment_name: str = "audio_spectrogram_classification"
    run_name: Optional[str] = None
    tracking_uri: str = "http://localhost:5000"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}