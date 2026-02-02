# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import librosa
from PIL import Image
import json
import copy

class AudioSpectrogramDataset(Dataset):
    """Dataset for audio spectrogram classification with metadata support"""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        mode: str = "train",
        metadata_file: Optional[str] = None,
        config: Optional[Any] = None
    ):
        #self.data_dir = Path(data_dir)
        self.data_dir = Path(data_dir).expanduser() #linux path
        self.transform = transform
        self.mode = mode
        self.config = config
        
        # Load data paths and labels
        self.samples = self._load_samples()
        
        # Load metadata if available
        self.metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def _load_samples(self) -> list:
        """Load spectrogram file paths and corresponding labels"""
        samples = []
        # Find all png files
        files = sorted(list(self.data_dir.glob("*.png")))
        
        # Extract classes from filenames (assuming format ClassName_...)
        classes =  sorted(list(set(f.name.split('_')[0] for f in files)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.classes = classes
        
        for img_file in files:
            class_name = img_file.name.split('_')[0]
            label = self.class_to_idx[class_name]
            samples.append((str(img_file), label))
            
        print(f"Found {len(classes)} classes: {self.class_to_idx}")
        return samples
    
    def _load_metadata(self, file_path: str) -> torch.Tensor:
        """Extract metadata for a given sample"""
        if not self.metadata:
            return torch.zeros(10)  # Default metadata size
            
        file_id = Path(file_path).stem
        if file_id in self.metadata:
            meta = self.metadata[file_id]
            # Convert metadata to tensor
            return torch.tensor([
                meta.get("duration", 0),
                meta.get("sample_rate", 0),
                meta.get("max_amplitude", 0),
                meta.get("mean_amplitude", 0),
                meta.get("zero_crossing_rate", 0),
                meta.get("spectral_centroid", 0),
                meta.get("spectral_bandwidth", 0),
                meta.get("spectral_rolloff", 0),
                meta.get("mfcc_mean", 0),
                meta.get("mfcc_std", 0)
            ], dtype=torch.float32)
        return torch.zeros(10)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        # Load spectrogram image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Load metadata
        metadata = self._load_metadata(img_path)
        
        return image, label, metadata


class AudioAugmentation:
    """Audio-specific data augmentations"""
    
    def __init__(self, config: Any):
        self.config = config
        
        # Standard spectrogram augmentations
        self.train_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    



def get_data_loaders(
    config: Any,
    fold_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with optional k-fold splitting"""
    
    aug = AudioAugmentation(config)
    
    if fold_indices is None:
        # Simple train/val split
        dataset = AudioSpectrogramDataset(
            data_dir=config.data_dir,
            transform=aug.train_transform,
            mode="train",
            config=config
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Apply different transforms for validation
        val_dataset.dataset.transform = aug.val_transform
    else:
        # K-fold split
        train_idx, val_idx = fold_indices
        full_dataset = AudioSpectrogramDataset(
            data_dir=config.data_dir,
            transform=aug.train_transform,
            mode="train",
            config=config
        )
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Clone dataset to apply different transforms
        val_dataset.dataset = copy.deepcopy(full_dataset)
        val_dataset.dataset.transform = aug.val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader