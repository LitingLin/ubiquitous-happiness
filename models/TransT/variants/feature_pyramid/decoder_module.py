import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, self_attention_modules, x_merged_cross_attention_decoder):
        super(Decoder, self).__init__()
        if self_attention_modules is not None:
            self.self_attentions = nn.ModuleList(self_attention_modules)
        self.x_merged_cross_attention_decoder = x_merged_cross_attention_decoder

    def forward(self, z, x):
        if hasattr(self, 'self_attentions'):
            for self_attention in self.self_attentions:
                z, x, = self_attention(z, x)
        self.x_merged_cross_attention_decoder(x, torch.cat((z, x), dim=1))
