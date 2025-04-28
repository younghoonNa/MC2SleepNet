import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .SleepConv_pyco import SleepConv_pyco

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module with optional attention masking.
    """
    def __init__(self, d_head, sequence_length, identity_matrix=None, device='cuda'):
        super().__init__()
        self.scale = 1 / (d_head ** 0.5)
        self.identity_matrix = identity_matrix

        if identity_matrix is None:
            self.attention_matrix = nn.Parameter(torch.ones(sequence_length, sequence_length), requires_grad=False)
        else:
            self.attention_matrix = nn.Parameter(torch.eye(sequence_length), requires_grad=False)
            if identity_matrix > 1:
                self.attention_matrix = nn.Parameter(torch.zeros(sequence_length, sequence_length), requires_grad=False)
                for idx in range(sequence_length):
                    start = max(0, idx - (identity_matrix - 1))
                    end = min(sequence_length, idx + identity_matrix)
                    self.attention_matrix[idx, start:end] = 1

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)

        if self.identity_matrix is not None:
            scores = scores.masked_fill(self.attention_matrix == 0, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_probs, V)
        return context

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module supporting optional activation functions.
    """
    def __init__(self, i_hidden=256, d_hidden=256, n_head=8, dropout=0.1,
                 sequence_length=14, identity_matrix=None, activation_query=None,
                 activation_value=None, bias=False, device='cuda'):
        super().__init__()

        self.d_head = d_hidden // n_head
        self.n_head = n_head

        self.W_Q = nn.Linear(i_hidden, d_hidden, bias=bias)
        self.W_K = nn.Linear(i_hidden, d_hidden, bias=bias)
        self.W_V = nn.Linear(i_hidden, d_hidden, bias=bias)
        self.linear = nn.Linear(d_hidden, i_hidden, bias=bias)

        self.dropout = nn.Dropout(p=dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, sequence_length, identity_matrix, device)

        self.activation_query = activation_query
        self.activation_value = activation_value

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        if self.activation_query is not None:
            Q = self.activation_query(self.W_Q(Q))
            K = self.activation_query(self.W_K(K))
            V = self.activation_query(self.W_V(V))
        else:
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        context = self.scaled_dot_attn(Q, K, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)

        output = self.linear(context)
        if self.activation_value is not None:
            output = self.activation_value(output)
        output = self.dropout(output)

        return output

class PoswiseFeedForwardNet(nn.Module):
    """
    Position-wise Feed Forward Network with optional activations and dropout.
    """
    def __init__(self, i_hidden=256, d_hidden=256, f_hidden=512, dropout=0.1,
                 first_active=F.gelu, second_active=None, bias=False):
        super().__init__()

        self.fc1 = nn.Linear(i_hidden, f_hidden, bias=bias)
        self.fc2 = nn.Linear(f_hidden, i_hidden, bias=bias)

        self.first_active = first_active
        self.second_active = second_active
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        output = self.fc1(inputs)
        if self.first_active is not None:
            output = self.first_active(output)
        output = self.dropout(output)

        output = self.fc2(output)
        if self.second_active is not None:
            output = self.second_active(output)
        output = self.dropout(output)

        return output

class CrossEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with optional Cross-Attention support.
    """
    def __init__(self, i_hidden=256, d_hidden=256, f_hidden=512, n_head=8, dropout=0.1,
                 layer_norm_epsilon=1e-12, sequence_length=14, identity_matrix=None,
                 mq_active=None, mv_active=None, f_first_active=F.gelu, f_second_active=None,
                 layer_norm=True, layer_norm_first=False, cross_attention=False, device='cuda'):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            i_hidden=i_hidden, d_hidden=d_hidden, n_head=n_head,
            dropout=dropout, sequence_length=sequence_length,
            identity_matrix=identity_matrix, activation_query=mq_active,
            activation_value=mv_active, device=device
        )
        self.cross_attn = MultiHeadAttention(
            i_hidden=i_hidden, d_hidden=d_hidden, n_head=n_head,
            dropout=dropout, sequence_length=sequence_length,
            identity_matrix=identity_matrix, activation_query=mq_active,
            activation_value=mv_active, device=device
        ) if cross_attention else None

        self.pos_ffn = PoswiseFeedForwardNet(i_hidden, d_hidden, f_hidden, dropout, f_first_active, f_second_active)

        self.layer_norm1 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.layer_norm3 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon) if cross_attention else None

        self.layer_norm = layer_norm
        self.layer_norm_first = layer_norm_first

    def forward(self, x, cross_value=None):
        """
        Args:
            x (Tensor): Input tensor.
            cross_value (Tensor or None): Tensor for cross-attention (optional).
        """
        if self.layer_norm_first:
            residual = x
            x = self.layer_norm1(x)

            x = residual + self.self_attn(x, x, x)
            residual = x

            if cross_value is not None and self.cross_attn is not None:
                x = self.layer_norm3(x)
                x = residual + self.cross_attn(x, cross_value, cross_value)

            residual = x
            x = self.layer_norm2(x)
            x = residual + self.pos_ffn(x)
        else:
            x = self.self_attn(x, x, x) + x
            if self.layer_norm:
                x = self.layer_norm1(x)

            if cross_value is not None and self.cross_attn is not None:
                x = self.cross_attn(x, cross_value, cross_value) + x
                if self.layer_norm:
                    x = self.layer_norm3(x)

            x = self.pos_ffn(x) + x
            if self.layer_norm:
                x = self.layer_norm2(x)

        return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Layer: injects positional information into embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
