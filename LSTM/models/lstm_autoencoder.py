import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder architecture for time series anomaly detection
    in solar power grid sensor data.
    
    This model encodes time series sequences into a latent representation
    and then decodes them back, with anomalies detected via reconstruction error.
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        latent_dim, 
        sequence_length,
        num_layers=2,
        dropout=0.2
    ):
        """
        Initialize the LSTM Autoencoder
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (e.g., number of sensors)
        hidden_dim : int
            Dimension of the hidden state in the LSTM layers
        latent_dim : int
            Dimension of the latent space
        sequence_length : int
            Length of the input sequences
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability for regularization
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent representation
        self.fc_encode = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (from latent to sequence)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        
        # Decoder LSTM (reconstructing the sequence)
        self.decoder = nn.LSTM(
            input_size=1,  # Using 1 as we feed one step at a time
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """
        Encode the input sequence into latent representation
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
        """
        # Pass through encoder LSTM
        _, (hidden, _) = self.encoder(x)
        
        # Take the last hidden state from the last layer
        hidden = hidden[-1]
        
        # Project to latent space
        latent = self.fc_encode(hidden)
        
        return latent
    
    def decode(self, latent):
        """
        Decode from latent representation back to sequence
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
            
        Returns:
        --------
        torch.Tensor
            Reconstructed sequence of shape (batch_size, sequence_length, input_dim)
        """
        batch_size = latent.size(0)
        
        # Project latent vector to hidden dimension
        hidden = self.fc_decode(latent)
        
        # Prepare hidden and cell states for decoder LSTM
        h0 = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, self.sequence_length, 1, device=latent.device)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.sequence_length, self.input_dim, device=latent.device)
        
        # Sequentially decode
        decoder_output, (hidden_state, cell_state) = self.decoder(
            decoder_input, (h0, c0)
        )
        
        # Project to output dimension
        outputs = self.output_layer(decoder_output)
        
        return outputs
    
    def forward(self, x):
        """
        Forward pass through the autoencoder
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Reconstructed output of shape (batch_size, sequence_length, input_dim)
        """
        # Encode
        latent = self.encode(x)
        
        # Decode
        outputs = self.decode(latent)
        
        return outputs


class LSTMAutoencoderTrainer:
    """
    Trainer class for the LSTM Autoencoder
    
    Handles data preparation, training, evaluation, and anomaly detection
    """
    def __init__(
        self,
        model,
        sequence_length=24,
        batch_size=32,
        learning_rate=0.001,
        device=None
    ):
        """
        Initialize the LSTM Autoencoder Trainer
        
        Parameters:
        -----------
        model : LSTMAutoencoder
            The autoencoder model to train
        sequence_length : int
            Length of the input sequences
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimization
        device : str, optional
            Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Scaler for data normalization
        self.scaler = MinMaxScaler()
    
    def create_sequences(self, data):
        """
        Create sequences from the data for training
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        list
            List of sequences, each of length sequence_length
        """
        sequences = []
        n_samples = len(data)
        
        for i in range(n_samples - self.sequence_length + 1):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
            
        return np.array(sequences)
    
    def prepare_data(self, df, train_ratio=0.8):
        """
        Prepare and split the data for training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with sensor readings
        train_ratio : float
            Ratio of data to use for training (vs. validation)
            
        Returns:
        --------
        tuple
            DataLoaders for training and validation sets
        """
        # Convert to numpy
        data = df.values
        
        # Scale the data
        self.scaler.fit(data)
        data_scaled = self.scaler.transform(data)
        
        # Create sequences
        sequences = self.create_sequences(data_scaled)
        
        # Split into training and validation
        n_train = int(len(sequences) * train_ratio)
        
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:]
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(train_sequences)
        val_tensor = torch.FloatTensor(val_sequences)
        
        # Create datasets
        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch
        
        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
            
        Returns:
        --------
        float
            Average training loss for the epoch
        """
        self.model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Move to device
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Compute loss
            loss = torch.mean(self.criterion(outputs, data))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Parameters:
        -----------
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data
            
        Returns:
        --------
        float
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                # Move to device
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = torch.mean(self.criterion(outputs, data))
                
                # Update metrics
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, df, epochs=50, train_ratio=0.8, verbose=True):
        """
        Train the model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with sensor readings
        epochs : int
            Number of training epochs
        train_ratio : float
            Ratio of data to use for training
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict
            Training history (losses)
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(df, train_ratio)
        
        # Track metrics
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}')
        
        return history
    
    def compute_reconstruction_error(self, df):
        """
        Compute reconstruction error for the entire dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with sensor readings
            
        Returns:
        --------
        numpy.ndarray
            Reconstruction error for each sample and each feature
        """
        # Convert to numpy and scale
        data = df.values
        data_scaled = self.scaler.transform(data)
        
        # Create sequences
        sequences = self.create_sequences(data_scaled)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Compute reconstruction
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(data_tensor)
            
        # Compute error (mean squared error per feature)
        errors = torch.mean(self.criterion(reconstructions, data_tensor), dim=1)
        
        # Convert to numpy
        errors = errors.cpu().numpy()
        
        # Allocate array for sequence-level errors
        sequence_errors = np.zeros((len(df), df.shape[1]))
        
        # Populate errors (average across all sequences containing each point)
        for i in range(len(sequences)):
            for j in range(self.sequence_length):
                sequence_errors[i + j] += errors[i, j]
        
        # Normalize by the number of sequences each point appears in
        counts = np.zeros(len(df))
        for i in range(len(sequences)):
            for j in range(self.sequence_length):
                counts[i + j] += 1
        
        for i in range(len(df)):
            if counts[i] > 0:
                sequence_errors[i] /= counts[i]
        
        return sequence_errors
    
    def detect_anomalies(self, df, threshold_percentile=99):
        """
        Detect anomalies based on reconstruction error
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with sensor readings
        threshold_percentile : float
            Percentile to use for anomaly threshold
            
        Returns:
        --------
        tuple
            (reconstruction_errors, anomaly_mask, threshold)
        """
        # Compute reconstruction error
        errors = self.compute_reconstruction_error(df)
        
        # Compute threshold (per feature)
        thresholds = np.percentile(errors, threshold_percentile, axis=0)
        
        # Detect anomalies
        anomalies = errors > thresholds
        
        return errors, anomalies, thresholds
    
    def plot_results(self, df, true_anomalies=None, window_size=500, sensors=None):
        """
        Plot the results of anomaly detection
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with sensor readings
        true_anomalies : pandas.DataFrame, optional
            True anomaly labels
        window_size : int
            Size of the window to plot
        sensors : list, optional
            List of sensor indices to plot (defaults to all)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Get reconstruction errors and detected anomalies
        errors, detected_anomalies, thresholds = self.detect_anomalies(df)
        
        # Determine which sensors to plot
        if sensors is None:
            sensors = list(range(df.shape[1]))
        
        # Get sensor names
        sensor_names = df.columns
        
        # Create figure
        n_sensors = len(sensors)
        fig, axs = plt.subplots(n_sensors, 1, figsize=(15, 4 * n_sensors), sharex=True)
        
        # Ensure axs is always a list
        if n_sensors == 1:
            axs = [axs]
        
        # Plot each sensor
        for i, sensor_idx in enumerate(sensors):
            sensor_name = sensor_names[sensor_idx]
            
            # Get data for the current window
            window_data = df.iloc[:window_size, sensor_idx].values
            window_errors = errors[:window_size, sensor_idx]
            window_anomalies = detected_anomalies[:window_size, sensor_idx]
            
            # Plot sensor data
            axs[i].plot(window_data, label='Sensor data', color='blue')
            
            # Plot threshold
            axs[i].axhline(y=thresholds[sensor_idx], color='orange', linestyle='--',
                         label=f'Threshold (p{thresholds[sensor_idx]:.2f})')
            
            # Plot detected anomalies
            if np.any(window_anomalies):
                anomaly_indices = np.where(window_anomalies)[0]
                axs[i].scatter(anomaly_indices, window_data[anomaly_indices],
                             color='red', label='Detected anomalies')
            
            # Plot true anomalies if provided
            if true_anomalies is not None:
                true_window_anomalies = true_anomalies.iloc[:window_size, sensor_idx].values
                if np.any(true_window_anomalies):
                    true_indices = np.where(true_window_anomalies)[0]
                    axs[i].scatter(true_indices, window_data[true_indices],
                                 color='green', marker='x', label='True anomalies')
            
            # Add labels
            axs[i].set_title(f'Sensor: {sensor_name}')
            axs[i].set_ylabel('Value')
            axs[i].legend()
            
            # Add secondary axis for error
            ax2 = axs[i].twinx()
            ax2.plot(window_errors, color='purple', alpha=0.5, label='Reconstruction error')
            ax2.set_ylabel('Error', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
        
        # Add overall labels
        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        
        return fig

    def save_model(self, path):
        """
        Save the model to a file
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'latent_dim': self.model.latent_dim,
            'num_layers': self.model.num_layers
        }, path)
        
    @staticmethod
    def load_model(path, device=None):
        """
        Load a model from a file
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        device : str, optional
            Device to load the model on
            
        Returns:
        --------
        LSTMAutoencoderTrainer
            Loaded trainer
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model
        model = LSTMAutoencoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            latent_dim=checkpoint['latent_dim'],
            sequence_length=checkpoint['sequence_length'],
            num_layers=checkpoint['num_layers']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = LSTMAutoencoderTrainer(
            model=model,
            sequence_length=checkpoint['sequence_length'],
            device=device
        )
        
        # Load optimizer
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler
        trainer.scaler = checkpoint['scaler']
        
        return trainer