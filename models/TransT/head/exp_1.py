import torch
from torch import nn

from models.modules.mlp import MLP


class EXP1Head_WithRegBranch(nn.Module):
    def __init__(self, transformer_hidden_dim):
        super().__init__()
        self.classification = MLP(transformer_hidden_dim, transformer_hidden_dim, 1, 3)
        self.regression = MLP(transformer_hidden_dim, transformer_hidden_dim, 5, 3)

    def forward(self, input_):
        assert input_.shape[0] == 1
        input_ = input_[0]

        class_branch = self.classification(input_).sigmoid()
        regression_branch = self.regression(input_).sigmoid()
        class_branch = class_branch.transpose(1, 2)
        class_score = class_branch[:, 0, :]
        bounding_box = regression_branch[:, :, 0: 4]
        quality_assessment = regression_branch[:, :, 4]

        return class_score, bounding_box, quality_assessment


class EXP1Head_WithClassBranch(nn.Module):
    def __init__(self, transformer_hidden_dim):
        super().__init__()
        self.classification = MLP(transformer_hidden_dim, transformer_hidden_dim, 2, 3)
        self.regression = MLP(transformer_hidden_dim, transformer_hidden_dim, 4, 3)

    def forward(self, input_):
        '''

        Args:
            input_ (torch.Tensor): (S, N, L, C)

        Returns:

        '''
        assert input_.shape[0] == 1
        input_ = input_[0]

        class_branch = self.classification(input_).sigmoid()
        regression_branch = self.regression(input_).sigmoid()
        class_branch = class_branch.transpose(1, 2)
        class_score = class_branch[:, 0, :]
        bounding_box = regression_branch
        quality_assessment = class_branch[:, 1, :]

        return class_score, bounding_box, quality_assessment
