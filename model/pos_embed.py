import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, init: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.max_len  = max_len
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)     
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)                              
        )
        self.register_buffer("position", pos)     
        self.register_buffer("div_term", div)

    def forward(self, x):
        seq_len = x.size(1)
        pos = self.position[:seq_len]                                
        angle = self.alpha * pos * self.div_term                    

        pe = torch.zeros(seq_len, self.d_model, device=angle.device)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        return pe.unsqueeze(0)  