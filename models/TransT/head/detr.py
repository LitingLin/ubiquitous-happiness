from torch import nn
from models.modules.mlp import MLP


class DETRHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.class_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, input_):
        outputs_class = self.class_embed(input_)
        outputs_coord = self.bbox_embed(input_).sigmoid()

        return outputs_class[-1], outputs_coord[-1]
