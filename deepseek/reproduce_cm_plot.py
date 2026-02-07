
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names):
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

if __name__ == "__main__":
    # Create a confusion matrix with some 0 values and some high values
    cm = np.array([
        [2552, 1, 0, 47],
        [0, 2000, 0, 0],
        [0, 0, 1000, 0],
        [0, 0, 0, 3500]
    ])
    class_names = ['Healthy', 'IMBALANCE2', 'IMBALANCE3', 'SEAWEED']
    
    fig = plot_confusion_matrix(cm, class_names)
    fig.savefig('repro_cm.png')
    print("Saved repro_cm.png")
