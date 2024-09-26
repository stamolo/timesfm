import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.linear_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.linear_output = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_resid = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                     .view(1, 1, config.seq_len, config.seq_len))
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim

    def forward(self, x):
        B, T, E = x.size()
        q, k, v = self.linear_proj(x).chunk(3, dim=2)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_attn(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.dropout_resid(self.linear_output(y))
        return y
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            NewGELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
    
    def forward(self, x):
        for block in self.decoder:
            x = block(x)

        return x
