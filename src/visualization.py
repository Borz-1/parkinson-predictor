import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history):
    metrics = ["loss", "accuracy", "recall"]
    for metric in metrics:
        if metric in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history.get(f'val_{metric}', []), label=f'Val {metric}')
            plt.title(f"{metric.capitalize()} over epochs")
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()