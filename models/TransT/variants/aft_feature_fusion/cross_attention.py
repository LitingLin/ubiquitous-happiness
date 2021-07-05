import torch.nn as nn
from models.TransT.aft import AFTFull, AFTFull_Parameterized


class AFTCrossAttention(nn.Module):
    def __init__(self, dim, z_size, x_size, aft_type):
        super(AFTCrossAttention, self).__init__()
        if aft_type == 'AFT-Full':
            self.z_x_aft = AFTFull(dim, dim, z_size[0] * z_size[1], x_size[0] * x_size[1])
            self.x_z_aft = AFTFull(dim, dim, x_size[0] * x_size[1], z_size[0] * z_size[1])
        elif aft_type == 'AFTFull_Parameterized':
            self.z_x_aft = AFTFull_Parameterized(dim, dim, z_size[0] * z_size[1], x_size[0] * x_size[1])
            self.x_z_aft = AFTFull_Parameterized(dim, dim, x_size[0] * x_size[1], z_size[0] * z_size[1])
        else:
            raise NotImplementedError(f"{aft_type} Not implemented")

    def forward(self, z, x):
        z_ = self.z_x_aft(z, x)
        x_ = self.x_z_aft(x, z)
        return z_, x_


class AFTCrossAttentionDecoder(nn.Module):
    def __init__(self, dim, z_size, x_size, aft_type):
        super(AFTCrossAttentionDecoder, self).__init__()
        if aft_type == 'AFT-Full':
            self.x_z_aft = AFTFull(dim, dim, x_size[0] * x_size[1], z_size[0] * z_size[1])
        elif aft_type == 'AFTFull_Parameterized':
            self.x_z_aft = AFTFull_Parameterized(dim, dim, x_size[0] * x_size[1], z_size[0] * z_size[1])
        else:
            raise NotImplementedError(f"{aft_type} Not implemented")

    def forward(self, z, x):
        x_ = self.x_z_aft(x, z)
        return x_
