import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import shutil

# Add current directory to path
sys.path.append(os.getcwd())

from trainer import Trainer
from config import Config

def test_advanced_metrics():
    print("Setting up test environment...")
    
    # Clean up previous runs
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
    
    # Mock Config
    class MockConfig:
        def __init__(self):
            self.tracking_uri = "mlruns"
            self.experiment_name = "test_metrics"
            self.run_name = "test_run"
            self.optimizer = "adam"
            self.learning_rate = 0.001
            self.weight_decay = 0.0
            self.momentum = 0.9
            self.scheduler = None
            self.model_type = "cnn_mlp"
            self.epochs = 2
            self.early_stopping_patience = 5
            
        def to_dict(self):
            return self.__dict__
            
    config = MockConfig()
    
    # Dummy Model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3) # 3 classes
            
        def forward(self, x, metadata=None):
            return self.fc(x)
            
    model = DummyModel()
    device = torch.device('cpu')
    
    # Dummy Data
    X = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    dataset = TensorDataset(X, y, X) # Use X as metadata dummy
    loader = DataLoader(dataset, batch_size=4)
    
    class_names = ['ClassA', 'ClassB', 'ClassC']
    
    print("Initializing Trainer...")
    trainer = Trainer(config, model, device)
    
    print("Starting Training...")
    history = trainer.train(loader, loader, class_names)
    
    print("Training Complete.")
    print("Metrics in history:", history.keys())
    
    # Check if artifacts exist (we can't easily check mlflow content programmatically without client, 
    # but we can check if it didn't crash and keys are present)
    expected_keys = [
        'val_roc_auc', 'val_pr_auc', 'val_log_loss', 'val_brier_score',
        'val_mcc'
    ]
    
    missing_keys = [k for k in expected_keys if k not in history]
    if missing_keys:
        print(f"FAILED: Missing metrics: {missing_keys}")
    else:
        print("SUCCESS: All expected metrics are present in history.")
        
    # Check for summary CSV
    if os.path.exists("summary_metrics.csv"):
        print("SUCCESS: summary_metrics.csv generated.")
    else:
        print("FAILED: summary_metrics.csv not found.")

if __name__ == "__main__":
    try:
        test_advanced_metrics()
    except Exception as e:
        print(f"Test Failed with error: {e}")
        import traceback
        traceback.print_exc()
