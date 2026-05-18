import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class SAGA(nn.Module):
    def __init__(self, win_size, d_model, d_model_in, adj_tensor, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model_in, bias=False)
        self.W2 = nn.Linear(d_model_in, d_model, bias=False)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

        self.theta_r = nn.Parameter(torch.empty(win_size))
        self.theta_c = nn.Parameter(torch.empty(win_size))
        nn.init.xavier_uniform_(self.theta_r.unsqueeze(0))
        nn.init.xavier_uniform_(self.theta_c.unsqueeze(0))

        self.tau_projection = nn.Linear(win_size, 1)
        nn.init.xavier_uniform_(self.tau_projection.weight)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = dropout

        self.g = nn.Parameter(adj_tensor.clone(), requires_grad=True)
   
    def forward(self, x):
        B = x.size(0)

        h = self.W1(x).transpose(1, 2)
        raw_tau = self.tau_projection(h)
        tau = torch.sigmoid(raw_tau*3.0)+1e-5
        
        dynamic_g = self.g.unsqueeze(0).expand(B, -1, -1) / tau
        P_s = torch.softmax(dynamic_g, dim=-1)
        
        e = self.leaky_relu(
            torch.matmul(h, self.theta_r)[:, :, None] +
            torch.matmul(h, self.theta_c)[:, None, :]
        )/math.sqrt(h.size(-1))        
        e = torch.softmax(e, dim=-1)

        a = torch.sigmoid(self.g).unsqueeze(0).expand(B, -1, -1)*e
        denom = a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        a = a/denom   
        
        series_s = F.dropout(a, p=self.dropout, training=self.training)
        
        z = torch.bmm(series_s, h).transpose(1, 2)

        return self.W2(z), e, P_s

 
class SpatialAttentionLayer(nn.Module):
    def __init__(self, win_size, d_model, n_heads, adj_tensor, d_model_in, max_b):
        super(SpatialAttentionLayer, self).__init__()
        
        self.inner_attention = SAGA(win_size, d_model, d_model_in, adj_tensor)

    def forward(self, x):
        out, s_s, p_s = self.inner_attention(x)
        return out, s_s, p_s