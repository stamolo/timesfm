import torch
import torch.nn as nn
import torch.nn.functional as F
from timesfm.transformer import TransformerDecoder
import numpy as np

to_np = lambda x: x.detach().cpu().numpy()

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int,
                 dropout_prob: float = 0.0, layer_norm: bool = False):
        super(ResidualBlock, self).__init__()
        
        self.layer_norm = layer_norm

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            Swish()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims, output_dims),
            nn.Dropout(dropout_prob)
        )
        
        self.residual_layer = nn.Linear(input_dims, output_dims)

        if layer_norm:
            self.ln_layer = nn.LayerNorm(output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        hidden = self.hidden_layer(x)
        
        output = self.output_layer(hidden)
    
        residual = self.residual_layer(x)
        
        if self.layer_norm:
            return self.ln_layer(output + residual)
        else:
            return output + residual

class TimesFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))
        self.input_residual = ResidualBlock(config.patch_len, config.patch_len*2, config.embed_dim)
        self.transformer_decoder = TransformerDecoder(config)
        self.output_residual = ResidualBlock(config.embed_dim, config.embed_dim, config.patch_len)
        self.patch_len = config.patch_len
        self.device = config.device
        self.eps = 1.0e-8
        
    def forward(self, x):
        B, TP = x.shape
        assert TP % self.patch_len == 0
        x = x.view(B, -1, self.patch_len)

        tok = self.input_residual(x)
        pos = self.pos_emb[:,:tok.shape[1],:]
        
        x = self.transformer_decoder(tok + pos)
        x = self.output_residual(x)
        x = x.view(B, -1)
        # x = self.revin.denormalize(x)
        return x
    
    def infer(self, datas, batch=False):
        isinstance(datas, np.ndarray)
        with torch.no_grad():
            x = torch.tensor(datas, dtype=torch.float32, device=self.device)
            if not batch:
                x = x.unsqueeze(0)
            x = self.normalize(x)
            y = self.forward(x)
            y = self.denormalize(y)
            y = y[:,-self.patch_len:]
            if not batch:
                y = y.squeeze(0)
            y = to_np(y)
            return y
    
    def normalize(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=True) + self.eps).detach()
        x = (x - self.mean) / self.std
        return x

    def denormalize(self, x):
        x = (x * self.std) + self.mean
        return x

class TimesFMConfig:
    def __init__(self):
        self.seq_len = 16
        self.patch_len = 32
        self.embed_dim = 512
        self.dropout = 0.1
        self.n_layer = 2
        self.n_head = 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
if __name__ == "__main__":
    config = TimesFMConfig()
    config.device = 'cuda'
    timesfm = TimesFM(config)

    T = config.seq_len
    P = config.patch_len
    B = 64
    x = torch.rand(B, T*P)
    y = timesfm(x)
    print(y)
    print(y.shape)

    timesfm = timesfm.to(config.device)
    x = np.random.rand(T*P) * 100
    y = timesfm.infer(x)
    print(y)
    print(y.shape)


    