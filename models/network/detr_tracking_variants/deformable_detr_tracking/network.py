from torch import nn
from models.modules.mlp import MLP


class DeformableDETRTracking(nn.Module):
    def __init__(self, backbone, transformer, num_queries, backbone_output_layers):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #self.bbox_embed = MLP(hidden_dim * num_queries, hidden_dim, 4, 3)
        self.num_feature_levels = len(backbone_output_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if self.num_feature_levels > 1:
            input_proj_list = []
            for i in range(self.num_feature_levels):
                in_channels = backbone.num_channels_output[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels_output[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

    def reset_parameters(self):
        #nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        #nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.bbox_embed.reset_parameters()
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, z, x):
        features, masks, position_encs = self.backbone(z, x)
        if not isinstance(features, list):
            features = [features]
            masks = [masks]
            position_encs = [position_encs]

        srcs = []
        for i, feat in enumerate(features):
            srcs.append(self.input_proj[i](feat))

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references = self.transformer(srcs, masks, position_encs, query_embeds)

        outputs_coord = self.bbox_embed(hs[:, 0, :]).sigmoid()
        #outputs_coord = self.bbox_embed(hs.flatten(start_dim=1)).sigmoid()
        return outputs_coord
