
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden, top_k=20):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.top_k = top_k
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)

    def encode(self, x):
        return F.linear(x, self.W_enc, self.b_enc)

    def decode(self, z):
        return F.linear(z, self.W_dec, self.b_dec)

    def topk_gating(self, z):
        # z: (batch, d_hidden)
        if self.top_k >= z.shape[-1]:
            return z
        # Zero out all but top-k activations per sample
        topk_vals, topk_idx = torch.topk(z, self.top_k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1.0)
        return z * mask

    def forward(self, x):
        z = F.relu(self.encode(x))
        z_sparse = self.topk_gating(z)
        x_recon = self.decode(z_sparse)
        return x_recon