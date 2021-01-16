import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def reset_parameters(self, method='kaiming'):
        if method == 'kaiming':
            for module in self.layers:
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_in',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif method == 'xavier':
            for module in self.layers:
                nn.init.xavier_uniform_(module.weight, gain=1)
                nn.init.constant_(module.bias, 0)
        else:
            raise Exception

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.single_bbox_output = True
        if self.single_bbox_output:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        else:
            self.bbox_embed = MLP(hidden_dim * num_queries, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def reset_parameters(self, method='kaiming'):
        self.bbox_embed.reset_parameters(method)

        if method == 'kaiming':
            nn.init.kaiming_normal_(self.input_proj.weight,
                                    mode='fan_in',
                                    nonlinearity='relu')
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
            nn.init.constant_(self.input_proj.bias, 0)
        else:
            raise Exception

    def forward(self, samples):
        z, z_mask, x = samples
        z_feat, z_feat_mask, z_feat_pos, x_feat, x_feat_mask, x_feat_pos = self.backbone(z, z_mask, x)

        hs = self.transformer(self.query_embed.weight, self.input_proj(z_feat), self.input_proj(x_feat),
                              z_feat_mask, x_feat_mask, z_feat_pos, x_feat_pos)[0]

        if self.single_bbox_output:
            outputs_coord = self.bbox_embed(hs[:, 0, :]).sigmoid()
        else:
            outputs_coord = self.bbox_embed(hs.flatten(start_dim=1)).sigmoid()

        return outputs_coord
