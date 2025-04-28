import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .Transformer import *
from .classifiers import get_classifier

class OuterPart(nn.Module):
    """
    OuterPart Module:
    Conducts sequence-level training with cross-modal masking and self-supervised learning.
    Designed to perform pretraining and fine-tuning in MC2SleepNet.
    """

    def __init__(self, config):
        super(OuterPart, self).__init__()

        self.cfg = config
        self.last_dim = 128
        self.training_mode = config['training_params']['mode']

        # Patch shuffle for masking training (data augmentation)
        self.shuffle = PatchShuffle(ratio=0.25)

        # Positional encoding for sequence processing
        self.pos_embedding = PositionalEncoding(self.last_dim)

        # Classifier heads
        self.classifier = get_classifier(config)
        self.total_classifier = nn.Linear(self.last_dim * 2, 5)

        # Cross-Attention Layers for Signal and Spectrogram Features
        self.outer_signal_layers = nn.ModuleList([
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

        self.outer_spc_layers = nn.ModuleList([
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

        # Masking module for pretraining
        self.seq_length = int(self.cfg['dataset']['seq_len'])
        self.masking_ratio = 0.25
        self.masking_setting = Masking_setting(
            emb_dim=self.last_dim,
            seq_length=self.seq_length,
            num_layers=1,
            num_heads=4
        )

    def forward(self, x, y):
        """
        Forward pass for OuterPart.

        Args:
            x (torch.Tensor): Signal modality tensor.
            y (torch.Tensor): Spectrogram modality tensor.

        Returns:
            tuple: outputs including logits for signals, spectrograms, fusion, and reconstruction loss
        """
        output_sig, output_spc, out_middle_sig, out_middle_spc, total = [], [], [], [], []
        recon_loss = torch.tensor(0).to(x.device)

        if self.training_mode == "pretrain":
            # Apply patch shuffling for data corruption
            patches_x, _, backward_indexes_x = self.shuffle(x)
            patches_y, _, backward_indexes_y = self.shuffle(y)

            # Mask signals and spectrograms separately
            corrupted_sig_out, normal_spc_out = self.masking_setting(patches_x, y, backward_indexes_x)
            corrupted_spc_out, normal_sig_out = self.masking_setting(patches_y, x, backward_indexes_y)

            final_sig_out = self.pos_embedding(x)
            final_spc_out = self.pos_embedding(y)

            # Cross-mask corrupted inputs
            for layer in self.outer_signal_layers:
                corrupted_sig_out = layer(corrupted_sig_out, normal_spc_out)

            for layer in self.outer_spc_layers:
                corrupted_spc_out = layer(corrupted_spc_out, normal_sig_out)

            # Process normal features without corruption (alignment training)
            for layer in self.outer_signal_layers:
                sig_out = layer(final_sig_out, None)

            for layer in self.outer_spc_layers:
                spc_out = layer(final_spc_out, None)

            # Compute reconstruction loss (L1 loss)
            L1_sig = F.l1_loss(x, corrupted_sig_out, reduction='sum')
            L1_spc = F.l1_loss(y, corrupted_spc_out, reduction='sum')
            recon_loss = (L1_sig + L1_spc) / 2

            # Classification heads for output
            for i in range(sig_out.size(1)):
                output_sig.append(self.classifier(sig_out[:, i, :]))
                output_spc.append(self.classifier(spc_out[:, i, :]))
                total.append(self.total_classifier(torch.cat([sig_out[:, i, :], spc_out[:, i, :]], dim=1)))

            output_sig = torch.stack(output_sig, dim=1)
            output_spc = torch.stack(output_spc, dim=1)
            total = torch.stack(total, dim=1)

        else:
            # Fine-tuning or normal supervised learning
            final_sig_out = self.pos_embedding(x)
            final_spc_out = self.pos_embedding(y)

            # Single-view encoding
            single_sig_out, single_spc_out = final_sig_out, final_spc_out
            for layer in self.outer_signal_layers:
                single_sig_out = layer(single_sig_out, None)
            for layer in self.outer_spc_layers:
                single_spc_out = layer(single_spc_out, None)

            # Multi-view cross-modal encoding
            sig_out, spc_out = final_sig_out, final_spc_out
            for layer in self.outer_signal_layers:
                sig_out = layer(sig_out, final_spc_out)
            for layer in self.outer_spc_layers:
                spc_out = layer(spc_out, final_sig_out)

            # Apply classifiers
            for i in range(final_spc_out.size(1)):
                output_sig.append(self.classifier(sig_out[:, i, :]))
                out_middle_sig.append(self.classifier(single_sig_out[:, i, :]))
                out_middle_spc.append(self.classifier(single_spc_out[:, i, :]))
                output_spc.append(self.classifier(spc_out[:, i, :]))
                total.append(self.total_classifier(torch.cat([sig_out[:, i, :], spc_out[:, i, :]], dim=1)))

            output_sig = torch.stack(output_sig, dim=1)
            output_spc = torch.stack(output_spc, dim=1)
            out_middle_sig = torch.stack(out_middle_sig, dim=1)
            out_middle_spc = torch.stack(out_middle_spc, dim=1)
            total = torch.stack(total, dim=1)

        return total, output_sig, output_spc, out_middle_sig, out_middle_spc, recon_loss
