"""
LSTM Autoencoder for anomaly detection in solar power grid sensor data.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder part of the LSTM autoencoder.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final FC layer to produce latent representation
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        # LSTM forward pass
        _, (hidden, _) = self.lstm(x)
        
        # Get the hidden state of the last LSTM layer
        hidden = hidden[-1]
        
        # Generate latent representation
        latent = self.fc(hidden)
        
        return latent


class Decoder(nn.Module):
    """
    Decoder part of the LSTM autoencoder.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # Transform latent vector to initial hidden state
        self.fc = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, latent):
        """
        Forward pass through the decoder.
        
        Args:
            latent: Latent representation of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed sequence of shape [batch_size, seq_len, output_dim]
        """
        batch_size = latent.size(0)
        
        # Transform latent to hidden state
        hidden = self.fc(latent)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        
        # Initialize with zeros as the first input
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=latent.device)
        
        # To store all outputs
        outputs = []
        
        # Decode step by step
        for _ in range(self.seq_len):
            out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            out = self.output_layer(out)
            outputs.append(out)
            decoder_input = out
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)
        
        return outputs


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for anomaly detection.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Encoder and decoder components
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Reconstructed sequence of shape [batch_size, seq_len, input_dim]
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed
    
    def get_reconstruction_error(self, x):
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Reconstruction error per sample
        """
        # Get reconstruction
        x_reconstructed = self.forward(x)
        
        # Compute MSE
        mse = ((x - x_reconstructed) ** 2).mean(dim=(1, 2))
        
        return mse