import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import *

class TransformerOuter(nn.Module):
    """
    TransformerOuter Module: 
    Performs cross-attention-based fusion between two different modality feature sets.
    Used in MC2SleepNet for sequence-level feature refinement.
    """

    def __init__(self, config):
        super(TransformerOuter, self).__init__()

        # Set training mode and model dimension
        self.training_mode = config['training_params']['mode']
        self.last_dim = config['classifier']['model_dim']

        # If using concatenated features (mixed layers), double the dimension
        if config['classifier']['mix_layer_feature']:
            self.last_dim *= 2

        # Define a list of Cross-Encoder Layers (Cross Attention + Self Attention)
        self.spc_cross_layers = nn.ModuleList([
            Cross_EncoderLayer(
                i_hidden=self.last_dim,
                d_hidden=128,
                f_hidden=1024,
                dropout=0.1,
                n_head=8,
                identity_matrix=None,
                sequence_length=16,
                mq_active=None,
                mv_active=None,
                f_first_active=F.gelu,
                f_second_active=None,
                layer_norm=True,
                layer_norm_first=False
            ) for _ in range(4)
        ])

        # Positional Encoding for temporal information
        self.pos_embedding = PositionalEncoding(self.last_dim)

    def forward(self, x, y):
        """
        Forward pass through TransformerOuter.

        Args:
            x (torch.Tensor): Signal modality feature tensor (batch_size, seq_len, feature_dim).
            y (torch.Tensor): Spectrogram modality feature tensor (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Fused feature representation after cross attention layers.
        """
        # Apply positional encoding to inputs
        sig_out = self.pos_embedding(x)

        if y is not None:
            spc_out = self.pos_embedding(y)
        else:
            spc_out = y  # In case of single modality input

        # Sequentially apply cross-attention layers
        for layer in self.spc_cross_layers:
            sig_out = layer(sig_out, spc_out)

        return sig_out
