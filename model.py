import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) with values from 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension by unsqueezing at dim 0(1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = (x + self.pe[:, :x.size(1), :]).require_grad_(False)
        x = self.dropout(x)
        return x
    
class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        return x