from torch import nn
from models.modules.mlp import MLP


class DETRHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classification = MLP(hidden_dim, hidden_dim, 2, 3)
        self.regression = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, input_):
        assert input_.shape[0] == 1
        input_ = input_[0]

        class_branch = self.classification(input_)
        regression_branch = self.regression(input_).sigmoid()
        class_branch = class_branch.transpose(1, 2)
        return class_branch, regression_branch
