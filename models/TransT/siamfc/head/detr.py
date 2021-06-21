import torch
from torch import nn
from models.modules.mlp import MLP


class DETRHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.classification = MLP(input_dim, hidden_dim, 2, 3)
        self.regression = MLP(input_dim, hidden_dim, 4, 3)

    def forward(self, cls_path, reg_path):
        '''
        Args:
            cls_path (torch.Tensor): (N, C, H, W)
            reg_path (torch.Tensor): (N, C, H, W)
        '''
        cls_path = cls_path.flatten(2).transpose(1, 2)  # (N, H*W, C)
        reg_path = reg_path.flatten(2).transpose(1, 2)  # (N, H*W, C)

        class_branch = self.classification(cls_path)
        regression_branch = self.regression(reg_path).sigmoid()
        class_branch = class_branch.transpose(1, 2)
        return class_branch, regression_branch
