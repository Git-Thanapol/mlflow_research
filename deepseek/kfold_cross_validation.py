# kfold_cross_validation.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from typing import List, Tuple, Dict
from config import Config
from models import get_model
from data.dataset import get_data_loaders
from trainer import Trainer
import warnings
import mlflow
import copy
warnings.filterwarnings('ignore')


class KFoldCrossValidator:
    """K-Fold Cross Validation orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kfold = StratifiedKFold(
            n_splits=config.k_folds,
            shuffle=True,
            random_state=config.seed
        )
        
    def _get_labels(self) -> np.ndarray:
        """Extract labels for stratification"""
        from data.dataset import AudioSpectrogramDataset
        
        dataset = AudioSpectrogramDataset(
            data_dir=self.config.data_dir,
            transform=None,
            config=self.config
        )
        
        labels = []
        for idx in range(len(dataset)):
            _, label, _ = dataset[idx]
            labels.append(label)
        
        return np.array(labels)
    
    
    def run(self, class_names: List[str]) -> Dict[str, List[float]]:
        """Run k-fold cross validation"""
        
        labels = self._get_labels()
        fold_results = []
        all_histories = []
        
        print(f"Starting {self.config.k_folds}-fold Cross Validation")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model_type}")
        
        # Track best model across all folds
        global_best_acc = 0
        global_best_model_state = None
        global_best_fold = -1
        
        # Start Parent Run for the entire experiment
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        experiment_run_name = f"{self.config.model_type}_Experiment"
        with mlflow.start_run(run_name=experiment_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            print(f"Started Parent MLflow Run: {experiment_run_name} (ID: {parent_run_id})")
            
            # Log common params to parent run
            mlflow.log_params(self.config.to_dict())
            
            for fold, (train_idx, val_idx) in enumerate(self.kfold.split(np.zeros(len(labels)), labels)):
                print(f"\n{'='*50}")
                print(f"Fold {fold + 1}/{self.config.k_folds}")
                print(f"{'='*50}")
                
                # Get data loaders for this fold
                train_loader, val_loader = get_data_loaders(
                    self.config,
                    fold_indices=(train_idx, val_idx)
                )
                
                # Initialize model
                model = get_model(self.config)
                
                # Initialize trainer
                trainer = Trainer(self.config, model, self.device)
                
                # Train and validate (will log nested runs)
                history = trainer.train(
                    train_loader,
                    val_loader,
                    class_names,
                    fold_idx=fold,
                    parent_run_id=parent_run_id
                )
                
                # Check if this fold produced the best model
                fold_best_acc = max(history['val_accuracy'])
                if fold_best_acc > global_best_acc:
                    global_best_acc = fold_best_acc
                    global_best_model_state = copy.deepcopy(model.state_dict())
                    global_best_fold = fold
                
                fold_results.append({
                    'fold': fold,
                    'best_val_accuracy': fold_best_acc,
                    'best_val_f1': max(history['val_f1']),
                    'history': history
                })
                
                all_histories.append(history)
            
            # Sub-step: Aggregate results
            self._aggregate_results(fold_results)
            
            # Sub-step: Register Best Model
            if global_best_model_state is not None:
                print(f"\n{'='*50}")
                print(f"Registering Best Model (from Fold {global_best_fold+1}, Acc: {global_best_acc:.4f})")
                
                # Need to recreate model architecture to load state dict
                best_model = get_model(self.config)
                best_model.load_state_dict(global_best_model_state)
                
                # Log best model to PARENT run and register it
                # User requested to "name a model after run name"
                model_name = experiment_run_name
                mlflow.pytorch.log_model(
                    best_model, 
                    artifact_path="best_model",
                    registered_model_name=model_name
                )
                print(f"Model registered as '{model_name}'")
        
        return {
            'fold_results': fold_results,
            'all_histories': all_histories
        }
    
    def _aggregate_results(self, fold_results: List[Dict]):
        """Aggregate and print k-fold results"""
        print(f"\n{'='*50}")
        print(f"K-Fold Cross Validation Results")
        print(f"{'='*50}")
        
        accuracies = [r['best_val_accuracy'] for r in fold_results]
        f1_scores = [r['best_val_f1'] for r in fold_results]
        
        print(f"\nPer-fold Validation Accuracy:")
        for i, acc in enumerate(accuracies):
            print(f"  Fold {i+1}: {acc:.4f}")
        
        print(f"\nPer-fold F1-Score (weighted):")
        for i, f1 in enumerate(f1_scores):
            print(f"  Fold {i+1}: {f1:.4f}")
        
        print(f"\nAggregated Results:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"  Min Accuracy: {np.min(accuracies):.4f}")
        print(f"  Max Accuracy: {np.max(accuracies):.4f}")
        
        
        # Remove these lines as logging is now done in run()
        # with mlflow.start_run(run_name=f"{self.config.model_type}_kfold_summary"):
        
        # Log aggregated metrics to the ACTIVE parent run (which is active when this is called)
        mlflow.log_metric("mean_accuracy", np.mean(accuracies))
        mlflow.log_metric("std_accuracy", np.std(accuracies))
        mlflow.log_metric("mean_f1", np.mean(f1_scores))
        mlflow.log_metric("std_f1", np.std(f1_scores))
        mlflow.log_metric("min_accuracy", np.min(accuracies))
        mlflow.log_metric("max_accuracy", np.max(accuracies))