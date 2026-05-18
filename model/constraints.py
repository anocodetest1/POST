import torch
import torch.nn as nn


class TripletLossWithKL(nn.Module):
    def __init__(self, B, margin=0.1, device='cuda'):
        super(TripletLossWithKL, self).__init__()
        self.B = B
        self.margin = margin
        self.device = device
        if margin == 0.:
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        self.neg_indices = self._generate_neg_indices(B, device)

    def _generate_neg_indices(self, B, device):
        neg_indices = torch.empty((B, B-1), dtype=torch.long, device=device)
        for b in range(B):
            indices = torch.cat((torch.arange(b, device=device), torch.arange(b+1, B, device=device)))
            neg_indices[b] = indices
        return neg_indices

    def symmetric_kl_divergence_batch(self, p, q, eps=1e-10):
        p = p + eps
        q = q + eps
        kl_pq = torch.sum(p * (p.log() - q.log()), dim=-1)
        kl_qp = torch.sum(q * (q.log() - p.log()), dim=-1)
        return kl_pq + kl_qp

    def forward(self, series):
        total_loss = 0.0
        for layer_series in series:
            B, H, N, _ = layer_series.shape
            agg_pdf = layer_series.mean(dim=2) 

            agg_pdf_flat = agg_pdf.view(B * H, N)
            p = agg_pdf_flat.unsqueeze(1) 
            q = agg_pdf_flat.unsqueeze(0)  
            kl_sym = self.symmetric_kl_divergence_batch(p, q)  
            kl_sym = kl_sym.view(B, H, B, H) 

            rand_indices = torch.randint(0, B-1, (B, H), device=self.device)  
            b_primes = self.neg_indices[torch.arange(B)[:, None], rand_indices]  

            dist_ap, dist_an = [], []
            for b in range(B):
                kl_to_other_heads = kl_sym[b, :, b, :].clone()  
                kl_to_other_heads[torch.arange(H, device=self.device), torch.arange(H, device=self.device)] = float('inf')
                d_pos = torch.min(kl_to_other_heads, dim=1)[0] 

                h_idx = torch.arange(H, device=self.device)
                d_neg = kl_sym[b, h_idx, b_primes[b], h_idx] 
                dist_ap.append(d_pos)
                dist_an.append(d_neg)

            if dist_ap:
                dist_ap = torch.cat(dist_ap)  
                dist_an = torch.cat(dist_an) 
                y = torch.ones_like(dist_an)
                if self.margin == 0.:
                    layer_loss = self.ranking_loss(dist_an - dist_ap, y)
                else:
                    layer_loss = self.ranking_loss(dist_an, dist_ap, y)
                total_loss += layer_loss

        return total_loss / len(series) if series else torch.tensor(0.0, device=self.device)