import torch.nn.functional as F
from torch import nn
import numpy as np
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * (d_model ** -0.5))
        self.d_model = d_model

    def forward(self, x):
        return self.weight[x] * (self.d_model ** 0.5)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for k in range(d_model // 2):
                freq = pos / (10000 ** (2*k / d_model))
                pe[pos, 2*k] = np.sin(freq)
                pe[pos, 2*k+1] = np.cos(freq)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, seq_len, n_heads, dropout = 0.1, masked = False):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model) 
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        self.masked = masked

    def forward(self, x, cross=None):
        B, S, D = x.shape
        cross = cross if cross is not None else x
        Q = self.W_Q(x)
        K = self.W_K(cross)
        V = self.W_V(cross)
        
        # want B, n_heads, S, d_k
        Q = Q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, cross.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, cross.size(1), self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if self.masked:
            scores = scores.masked_fill(self.mask[:S, :S], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V
        # back to B, S, D
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_dropout(self.W_O(out))

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout = 0.1, d_ff = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.source_MHA = MultiHeadAttention(d_model, max_seq_len, n_heads, dropout=dropout, masked=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.source_MHA(self.norm1(x))
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout = 0.1, d_ff = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.masked_MHA = MultiHeadAttention(d_model, max_seq_len, n_heads, dropout=dropout, masked=True)
        self.cross_MHA = MultiHeadAttention(d_model, max_seq_len, n_heads, dropout=dropout, masked=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, target, cross):
        target = target + self.masked_MHA(self.norm1(target))
        target = target + self.cross_MHA(self.norm2(target), cross)
        target = target + self.ff_dropout(self.ff(self.norm3(target)))
        return target


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, n_heads, source_l_layers, target_l_layers, max_seq_len, dropout = 0.1):
        super().__init__()
        self.target_embed = Embedding(target_vocab_size, d_model)
        self.source_embed = self.target_embed if source_vocab_size == target_vocab_size else Embedding(source_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([Encoder(d_model, n_heads, max_seq_len, dropout=dropout) for _ in range(source_l_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(d_model, n_heads, max_seq_len, dropout=dropout) for _ in range(target_l_layers)])
        self.output_proj = nn.Linear(d_model, target_vocab_size, bias=False)
        self.output_proj.weight = self.target_embed.weight
    
    def encoder(self, source):
        source = self.embed_dropout(self.pos_enc(self.source_embed(source)))
        for layer in self.encoder_layers:
            source = layer(source)
        return source

    def decoder(self, target, enc_out):
        target = self.embed_dropout(self.pos_enc(self.target_embed(target)))
        for layer in self.decoder_layers:
            target = layer(target, enc_out)
        logits = self.output_proj(target)
        return logits

    def forward(self, source, target):
        enc_out = self.encoder(source)
        dec_out = self.decoder(target, enc_out)
        return dec_out