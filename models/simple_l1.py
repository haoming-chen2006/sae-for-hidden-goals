import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoderL1(nn.Module):
    def __init__(self, d_in, d_hidden, l1_lambda=1e-3):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.l1_lambda = l1_lambda
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)

    def encode(self, x):
        return F.linear(x, self.W_enc, self.b_enc)

    def decode(self, z):
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x):
        z = F.relu(self.encode(x))
        self.sparsity_loss = self.l1_lambda * torch.mean(torch.abs(z))
        x_recon = self.decode(z)
        return x_recon

    def get_sparsity_loss(self):
        return getattr(self, 'sparsity_loss', 0.0)