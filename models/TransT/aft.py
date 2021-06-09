import torch
from torch import nn


class AFTFull(nn.Module):
    def __init__(self, dim, hidden_dim, q_T, kv_T):
        super(AFTFull, self).__init__()
        self.qW = nn.Linear(dim, hidden_dim)
        self.kW = nn.Linear(dim, hidden_dim)
        self.vW = nn.Linear(dim, hidden_dim)
        self.w_bias = nn.Parameter(torch.zeros(q_T, kv_T))

    def forward(self, q, kv):
        q = self.qW(q)
        k = self.kW(kv)
        v = self.vW(kv)

        A = torch.exp(k.unsqueeze(-3) + self.w_bias.unsqueeze(-1))
        AV = A * v.unsqueeze(-3)
        b = AV.sum(-2)
        b = b / A.sum(-2)

        final = q.sigmoid()
        final = final * b
        return final


class AFTFull_Parameterized(nn.Module):
    def __init__(self, dim, hidden_dim, q_T, kv_T, bias_dim=128):
        super(AFTFull_Parameterized, self).__init__()
        self.qW = nn.Linear(dim, hidden_dim)
        self.kW = nn.Linear(dim, hidden_dim)
        self.vW = nn.Linear(dim, hidden_dim)
        self.w_bias_u = nn.Parameter(torch.zeros(q_T, bias_dim))
        self.w_bias_v = nn.Parameter(torch.zeros(bias_dim, kv_T))

    def forward(self, q, kv):
        q = self.qW(q)
        k = self.kW(kv)
        v = self.vW(kv)

        w_bias = self.w_bias_u @ self.w_bias_v
        A = torch.exp(k.unsqueeze(-3) + w_bias.unsqueeze(-1))
        AV = A * v.unsqueeze(-3)
        b = AV.sum(-2)
        b = b / A.sum(-2)

        final = q.sigmoid()
        final = final * b
        return final




if __name__ == '__main__':
    a = AFTFull_Parameterized(64,  64, 10, 12)
    b = torch.zeros([3, 10, 64])
    c = torch.zeros([3, 12, 64])
    a(b, c)
