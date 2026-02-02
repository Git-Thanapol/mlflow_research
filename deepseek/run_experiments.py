# run_experiments.py
import subprocess
import time
from typing import List

def run_all_experiments():
    """Run experiments for all model architectures"""
    
    models = ["baseline_cnn", "cnn_mlp", "cnn_attention", "ast"]
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Starting experiment for {model}")
        print(f"{'='*60}")
        
        # Run with k-fold cross validation
        cmd = [
            "python", "main.py",
            "--model", model,
            "--k_folds", "5",
            "--epochs", "50",
            "--batch_size", "32",
            "--lr", "1e-3"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        elapsed = time.time() - start_time
        print(f"Completed {model} in {elapsed:.2f} seconds")
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_all_experiments()