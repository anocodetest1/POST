import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn_t import AnomalyAttention, AttentionLayer
from .attn_s import SpatialAttentionLayer
from .embed import DataEmbedding
from utils.knn_graph_builder import KNNGraphBuilder


class EncoderLayer(nn.Module):
    def __init__(self, attention_t, attention_s, d_model, attn_s_in=False, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attn_t = attention_t
        self.attn_s = attention_s
        self.attn_s_in = attn_s_in

        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        if self.attn_s_in:
            x_new, series_s, p_s = self.attn_s(x)
            x = x + x_new

        new_x, attn, mask, sigma = self.attn_t(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma, series_s, p_s


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        series_s_list = []
        p_s_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma, series_s, p_s = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            series_s_list.append(series_s)
            p_s_list.append(p_s)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, series_s_list, p_s_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, data_path, dataset, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn_s_in=False, max_b=32, activation='gelu', is_single_coords=True, output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.H = n_heads
        self.is_single_coords = is_single_coords
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        
        g_builder = KNNGraphBuilder(k=5)
        adj = g_builder.build_from_file(data_path, dataset)
        adj_tensor = torch.from_numpy(adj)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size, n_heads, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    SpatialAttentionLayer(win_size, d_model, n_heads, adj_tensor, d_model_in=enc_in, max_b=max_b),
                    d_model, attn_s_in, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.gs = []
        for i in range(len(self.encoder.attn_layers)):
            self.gs.append(self.encoder.attn_layers[i].attn_s.inner_attention.g)

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        B, L, _ = enc_out.shape
        enc_out, series, prior, sigmas, series_s, p_ss = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:            
            return enc_out, series, prior, sigmas, series_s, p_ss
        else:
            return enc_out