import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from trainer import Trainer

# Mock Config
class MockConfig:
    def __init__(self):
        self.tracking_uri = "mlruns"
        self.experiment_name = "test_exp"
        self.optimizer = "adam"
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.momentum = 0.9
        self.scheduler = None
        self.model_type = "cnn_mlp"
        self.epochs = 1
        
config = MockConfig()
model = nn.Linear(10, 2) # Dummy model
device = torch.device('cpu')

print("Initializing Trainer...")
try:
    # Initialize Trainer
    trainer = Trainer(config, model, device)

    # Create dummy CM
    cm = np.array([
        [2552, 1, 0, 47],
        [0, 2000, 0, 0],
        [0, 0, 1000, 0],
        [0, 0, 0, 3500]
    ])
    class_names = ['Healthy', 'IMBALANCE2', 'IMBALANCE3', 'SEAWEED']

    print("Generating plot...")
    fig = trainer.plot_confusion_matrix(cm, class_names)
    output_file = "test_conf_matrix_fixed.png"
    fig.savefig(output_file)
    print(f"Saved {output_file}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
