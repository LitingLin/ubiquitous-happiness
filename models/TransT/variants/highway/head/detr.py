import torch.nn as nn
from models.modules.mlp import MLP


class DETRHead(nn.Module):
    def __init__(self, classification_hidden_dim, regression_hidden_dim):
        super().__init__()
        self.classification = MLP(classification_hidden_dim, classification_hidden_dim, 2, 3)
        self.regression = MLP(regression_hidden_dim, regression_hidden_dim, 4, 3)

    def forward(self, classification_branch, regression_branch):
        assert classification_branch.shape[0] == 1 and regression_branch.shape[0] == 1
        classification_branch = classification_branch[0]
        regression_branch = regression_branch[0]

        class_branch = self.classification(classification_branch)
        regression_branch = self.regression(regression_branch).sigmoid()
        class_branch = class_branch.transpose(1, 2)
        return class_branch, regression_branch
