# main.py
import argparse
import torch
import numpy as np
import random
from typing import List
from config import Config
from kfold_cross_validation import KFoldCrossValidator
from models import get_model
from data.dataset import get_data_loaders
from trainer import Trainer

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Audio Spectrogram Classification Framework")
    parser.add_argument("--model", type=str, default="baseline_cnn",
                       choices=["baseline_cnn", "cnn_mlp", "cnn_attention", "ast"],
                       help="Model architecture to use")
    parser.add_argument("--k_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data/spectrograms",
                       help="Directory containing spectrogram data")
    parser.add_argument("--no_kfold", action="store_true",
                       help="Disable k-fold cross validation")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = Config(
        model_type=args.model,
        k_folds=args.k_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir
    )
    
    # Class names (modify based on your dataset)
    class_names = [f"class_{i}" for i in range(config.num_classes)]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.no_kfold or config.k_folds <= 1:
        # Single train/val split
        print(f"\nTraining {config.model_type} with single split")
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(config)
        
        # Initialize model
        model = get_model(config)
        
        # Initialize trainer
        trainer = Trainer(config, model, device)
        
        # Train
        history = trainer.train(train_loader, val_loader, class_names)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        
    else:
        # K-Fold Cross Validation
        print(f"\nRunning {config.k_folds}-fold cross validation for {config.model_type}")
        
        validator = KFoldCrossValidator(config)
        results = validator.run(class_names)
        
        print(f"\nCross-validation completed!")

if __name__ == "__main__":
    main()