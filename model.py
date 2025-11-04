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
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1 in paper
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2 in paper
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        return x

class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(MultiheadAttentionBlock, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
      
    @staticmethod
    def attention(Q, K, V, mask=None, dropout=None):
        d_k = Q.shape[-1]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        output = torch.matmul(attention_scores, V)
        return output, attention_scores

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        K = self.w_k(key)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        V = self.w_v(value) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) --> (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) --> (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) --> (batch_size, num_heads, seq_len, d_k)

        x, self.attention_scores = MultiheadAttentionBlock.attention(Q, K, V, mask=mask, dropout=self.dropout_layer)
        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = self.w_o(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return x
    
class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super(ResidualConnection, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        # Apply layer normalization, then the sublayer (e.g., attention or feed-forward), then dropout, and add the original input (residual connection)
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(self_attention_block.d_model,dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, num_layers: int, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(self_attention_block, feed_forward_block, dropout) for _ in range(num_layers)])
        self.layer_norm = LayerNormalization(self_attention_block.d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, num_layers: int, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout) for _ in range(num_layers)])
        self.layer_norm = LayerNormalization(self_attention_block.d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.layer_norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, seq_len: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super(Transformer, self).__init__()
        self.src_embedding = InputEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(seq_len, d_model, dropout)

        self.self_attention_block = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.cross_attention_block = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        self.encoder = Encoder(num_layers, self.self_attention_block, self.feed_forward_block, dropout)
        self.decoder = Decoder(num_layers, self.self_attention_block, self.cross_attention_block, self.feed_forward_block, dropout)

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        src_embedded = self.positional_encoding(self.src_embedding(src))
        enc_output = self.encoder(src_embedded, src_mask)

        # Decode
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.projection_layer(dec_output)
        return output
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, seq_len: int = 512, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1) -> Transformer:
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    )