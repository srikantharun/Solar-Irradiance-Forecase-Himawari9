"""
Visualization utilities for solar grid anomaly detection.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_time_series(df, sensor_cols, anomaly_col='anomaly', figsize=(12, 8)):
    """
    Plot time series data with anomalies highlighted.
    
    Args:
        df: DataFrame with time series data
        sensor_cols: List of sensor columns to plot
        anomaly_col: Column indicating anomalies
        figsize: Figure size
    """
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=figsize, sharex=True)
    
    if len(sensor_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(sensor_cols):
        # Plot sensor data
        axes[i].plot(df.index, df[col], label=col, color='blue', alpha=0.7)
        
        # Highlight anomalies
        anomaly_points = df[df[anomaly_col] == 1]
        if not anomaly_points.empty:
            axes[i].scatter(anomaly_points.index, anomaly_points[col], 
                          color='red', marker='o', label='Anomalies', s=25)
            
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.suptitle('Solar Grid Sensor Data with Anomalies', y=1.02)
    
    return fig


def plot_reconstruction(original, reconstructed, idx=0, figsize=(12, 6)):
    """
    Plot original vs reconstructed sequences.
    
    Args:
        original: Original sequences
        reconstructed: Reconstructed sequences
        idx: Index of sequence to plot
        figsize: Figure size
    """
    seq_len, n_features = original.shape[1:]
    
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    x = np.arange(seq_len)
    
    for i in range(n_features):
        axes[i].plot(x, original[idx, :, i], 'b-', label='Original')
        axes[i].plot(x, reconstructed[idx, :, i], 'r--', label='Reconstructed')
        axes[i].set_ylabel(f'Feature {i+1}')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.suptitle('Original vs Reconstructed Sequence', y=1.02)
    
    return fig


def plot_reconstruction_error(error, threshold=None, anomaly_indices=None, figsize=(12, 4)):
    """
    Plot reconstruction error with threshold.
    
    Args:
        error: Reconstruction error for each sample
        threshold: Anomaly threshold
        anomaly_indices: Indices of true anomalies
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(error, 'b-', alpha=0.7, label='Reconstruction Error')
    
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    if anomaly_indices is not None:
        plt.scatter(anomaly_indices, error[anomaly_indices], 
                   color='red', marker='o', label='True Anomalies')
    
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Reconstruction Error')
    plt.tight_layout()
    
    return plt.gcf()


def plot_evaluation_metrics(y_true, y_scores, figsize=(16, 4)):
    """
    Plot evaluation metrics: PR curve, ROC curve, and confusion matrix.
    
    Args:
        y_true: True labels
        y_scores: Anomaly scores
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    axes[0].plot(recall, precision, 'b-')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve (AUC: {pr_auc:.3f})')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, 'b-')
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve (AUC: {roc_auc:.3f})')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Determine optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = y_scores[optimal_idx] if len(y_scores) > optimal_idx else 0.5
    
    # Confusion matrix
    y_pred = (y_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[2])
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')
    axes[2].set_title(f'Confusion Matrix (Threshold: {optimal_threshold:.3f})')
    
    plt.tight_layout()
    
    return fig, optimal_threshold