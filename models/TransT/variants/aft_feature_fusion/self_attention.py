import torch.nn as nn
from models.TransT.aft import AFTFull, AFTFull_Parameterized


class AFTSelfAttention(nn.Module):
    def __init__(self, dim, size, aft_type):
        super(AFTSelfAttention, self).__init__()
        if aft_type == 'AFT-Full':
            self.aft = AFTFull(dim, dim, size[0] * size[1], size[0] * size[1])
        elif aft_type == 'AFTFull_Parameterized':
            self.aft = AFTFull_Parameterized(dim, dim, size[0] * size[1], size[0] * size[1])
        else:
            raise NotImplementedError(f"{aft_type} Not implemented")

    def forward(self, x):
        return self.aft(x, x)
