import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.manifold import TSNE

def plot_solar_data(df, sensors=None, days=7, anomalies=None, figsize=(14, 8)):
    """
    Plot solar power grid sensor data with optional anomaly markers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing sensor data with datetime index
    sensors : list, optional
        List of sensor columns to plot (defaults to all)
    days : int
        Number of days to plot
    anomalies : pandas.DataFrame, optional
        DataFrame with anomaly labels (same shape as df)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Select sensors to plot
    if sensors is None:
        sensors = df.columns.tolist()
    
    # Calculate number of data points to plot
    points_per_day = 24 * 60 // 15  # Assuming 15-min intervals
    n_points = days * points_per_day
    
    # Subset data for plotting
    plot_df = df.iloc[:n_points][sensors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each sensor
    for sensor in sensors:
        ax.plot(plot_df.index, plot_df[sensor], label=sensor, alpha=0.8)
    
    # Add anomaly markers if provided
    if anomalies is not None:
        anomaly_df = anomalies.iloc[:n_points][sensors]
        for sensor in sensors:
            # Find indices with anomalies
            anomaly_idx = anomaly_df.index[anomaly_df[sensor] == 1]
            if len(anomaly_idx) > 0:
                ax.scatter(
                    anomaly_idx, 
                    plot_df.loc[anomaly_idx, sensor],
                    color='red', 
                    marker='x', 
                    s=100, 
                    label=f'{sensor} anomalies' if sensor == sensors[0] else None
                )
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    
    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Solar Power Generation - {days} Days')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_daily_patterns(df, sensors=None, figsize=(14, 8)):
    """
    Plot average daily patterns for solar power sensors
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing sensor data with datetime index
    sensors : list, optional
        List of sensor columns to plot (defaults to all)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Select sensors to plot
    if sensors is None:
        sensors = df.columns.tolist()
    
    # Create hour of day feature
    hour_df = df.copy()
    hour_df['hour'] = hour_df.index.hour + hour_df.index.minute / 60
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each sensor
    for sensor in sensors:
        # Group by hour and calculate mean
        hourly_avg = hour_df.groupby('hour')[sensor].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, label=sensor, alpha=0.8)
    
    # Add labels and legend
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Average Daily Solar Power Generation Pattern')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_monthly_patterns(df, sensors=None, figsize=(14, 8)):
    """
    Plot average monthly patterns for solar power sensors
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing sensor data with datetime index
    sensors : list, optional
        List of sensor columns to plot (defaults to all)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Select sensors to plot
    if sensors is None:
        sensors = df.columns.tolist()
    
    # Create month feature
    month_df = df.copy()
    month_df['month'] = month_df.index.month
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each sensor
    for sensor in sensors:
        # Group by month and calculate mean
        monthly_avg = month_df.groupby('month')[sensor].mean()
        ax.plot(monthly_avg.index, monthly_avg.values, 
                marker='o', label=sensor, alpha=0.8)
    
    # Add labels and legend
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Average Monthly Solar Power Generation Pattern')
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_reconstruction_error(errors, threshold=None, figsize=(14, 6)):
    """
    Plot reconstruction error over time
    
    Parameters:
    -----------
    errors : numpy.ndarray
        Array of reconstruction errors (samples, features)
    threshold : float or array, optional
        Threshold for anomaly detection
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate mean error across features
    mean_error = np.mean(errors, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot error
    ax.plot(mean_error, label='Mean Reconstruction Error')
    
    # Add threshold line if provided
    if threshold is not None:
        if isinstance(threshold, (list, np.ndarray)):
            threshold = np.mean(threshold)
        ax.axhline(y=threshold, color='r', linestyle='--', 
                 label=f'Threshold: {threshold:.4f}')
    
    # Add labels and legend
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Reconstruction Error Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_reconstruction_error(errors, sensor_names=None, figsize=(14, 6)):
    """
    Plot reconstruction error by feature (sensor)
    
    Parameters:
    -----------
    errors : numpy.ndarray
        Array of reconstruction errors (samples, features)
    sensor_names : list, optional
        List of sensor names
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate mean error per feature
    mean_errors = np.mean(errors, axis=0)
    
    # Create labels
    if sensor_names is None:
        sensor_names = [f"Sensor {i+1}" for i in range(len(mean_errors))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    ax.bar(sensor_names, mean_errors)
    
    # Add labels
    ax.set_xlabel('Sensor')
    ax.set_ylabel('Average Reconstruction Error')
    ax.set_title('Average Reconstruction Error by Sensor')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_error_distribution(errors, figsize=(14, 6)):
    """
    Plot distribution of reconstruction errors
    
    Parameters:
    -----------
    errors : numpy.ndarray
        Array of reconstruction errors (samples, features)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate mean error across features
    mean_error = np.mean(errors, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distribution
    sns.histplot(mean_error, kde=True, ax=ax)
    
    # Add labels
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Reconstruction Errors')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_tsne_visualization(sequences, labels=None, figsize=(10, 8)):
    """
    Plot t-SNE visualization of sequence embeddings
    
    Parameters:
    -----------
    sequences : numpy.ndarray
        Array of sequence data (sequences, sequence_length, features)
    labels : numpy.ndarray, optional
        Array of labels (1 for anomaly, 0 for normal)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Reshape sequences for t-SNE
    n_sequences, seq_len, n_features = sequences.shape
    X_flat = sequences.reshape(n_sequences, seq_len * n_features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_flat)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                           cmap='coolwarm', alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(),
                          title="Classes")
        ax.add_artist(legend1)
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    
    # Add labels
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_title('t-SNE Visualization of Sequence Embeddings')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_anomaly_metrics(y_true, y_score, figsize=(16, 6)):
    """
    Plot ROC and Precision-Recall curves for anomaly detection
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True anomaly labels (1 for anomaly, 0 for normal)
    y_score : numpy.ndarray
        Anomaly scores (higher values indicate more likely anomalies)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    Plot confusion matrix for anomaly detection
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True anomaly labels (1 for anomaly, 0 for normal)
    y_pred : numpy.ndarray
        Predicted anomaly labels (1 for anomaly, 0 for normal)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Add labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    
    plt.tight_layout()
    return fig

def plot_training_history(history, figsize=(10, 6)):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training and validation loss
    ax.plot(history['train_loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    
    # Add labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig