# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    """Training and validation orchestrator with MLflow integration"""
    
    def __init__(self, config: Any, model: nn.Module, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.epoch = 0
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        
        # Scheduler
        if config.scheduler == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        elif config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1} Training")
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            metadata = metadata.to(self.device) if metadata is not None else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.model_type == "cnn_mlp":
                outputs = self.model(images, metadata)
            else:
                outputs = self.model(images)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'train_precision': precision,
            'train_recall': recall
        }
    
    def validate(self, val_loader) -> Tuple[Dict[str, float], np.ndarray]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                metadata = metadata.to(self.device) if metadata is not None else None
                
                if self.config.model_type == "cnn_mlp":
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> plt.Figure:
        """Create confusion matrix visualization"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        return fig
    
    def log_to_mlflow(self, params: Dict, metrics: Dict, cm_fig: Optional[plt.Figure] = None):
        """Log experiment data to MLflow"""
        with mlflow.start_run(run_name=self.config.run_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            if cm_fig:
                mlflow.log_figure(cm_fig, "confusion_matrix.png")
                plt.close(cm_fig)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Log additional artifacts
            if hasattr(self.config, 'to_dict'):
                import json
                with open("config.json", "w") as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                mlflow.log_artifact("config.json")
    
    def train(
        self,
        train_loader,
        val_loader,
        class_names: List[str],
        fold_idx: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Main training loop"""
        
        # Initialize tracking
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_f1': []
        }
        
        best_val_accuracy = 0
        best_model_state = None
        
        # MLflow run name
        if fold_idx is not None:
            self.config.run_name = f"{self.config.model_type}_fold_{fold_idx}"
        else:
            self.config.run_name = f"{self.config.model_type}_single_run"
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics, cm = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_accuracy'])
                else:
                    self.scheduler.step()
            
            # Update history
            for k in train_metrics:
                history[k].append(train_metrics[k])
            for k in val_metrics:
                history[k].append(val_metrics[k])
            
            # Log to console
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Accuracy: {train_metrics['train_accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_accuracy']
                best_model_state = self.model.state_dict().copy()
                
                # Plot confusion matrix for best epoch
                cm_fig = self.plot_confusion_matrix(cm, class_names)
                
                # Prepare metrics for MLflow
                all_metrics = {**train_metrics, **val_metrics}
                all_metrics['best_val_accuracy'] = best_val_accuracy
                all_metrics['epoch'] = epoch + 1
                
                # Log to MLflow
                params = self.config.to_dict()
                params['fold'] = fold_idx if fold_idx is not None else 0
                self.log_to_mlflow(params, all_metrics, cm_fig)
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return history