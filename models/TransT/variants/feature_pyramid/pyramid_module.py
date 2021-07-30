import torch
import torch.nn as nn


class PyramidModule(nn.Module):
    def __init__(self, level_0_self_attention_modules, level_1_self_attention_modules, level_0_1_z_cross_attention_module, level_0_1_x_cross_attention_module):
        super(PyramidModule, self).__init__()
        self.level_0_self_attentions = nn.ModuleList(level_0_self_attention_modules)
        self.level_1_self_attentions = nn.ModuleList(level_1_self_attention_modules)
        self.level_0_1_z_cross_attention = level_0_1_z_cross_attention_module
        self.level_0_1_x_cross_attention = level_0_1_x_cross_attention_module

    def forward(self,
                level_0_z, level_0_x, level_0_z_H, level_0_z_W, level_0_x_H, level_0_x_W,
                level_1_z, level_1_x, level_1_z_H, level_1_z_W, level_1_x_H, level_1_x_W,
                level_0_self_attention_q_positional_encodings, level_0_self_attention_k_positional_encodings,
                level_1_self_attention_q_positional_encodings, level_1_self_attention_k_positional_encodings,
                level_0_z_cross_attention_positional_encoding, level_0_x_cross_attention_positional_encoding,
                level_1_z_cross_attention_positional_encoding, level_1_x_cross_attention_positional_encoding):
        level_0_z_size = level_0_z_H * level_0_z_W
        level_0_x_size = level_0_x_H * level_0_x_W
        assert level_0_z_size == level_0_z.shape[1]
        assert level_0_x_size == level_0_x.shape[1]
        merged_0 = torch.cat((level_0_z, level_0_x), dim=1)
        for level_0_self_attention, level_0_self_attention_q_positional_encoding, level_0_self_attention_k_positional_encoding in zip(
                self.level_0_self_attentions, level_0_self_attention_q_positional_encodings, level_0_self_attention_k_positional_encodings):
            merged_0 = level_0_self_attention(merged_0, level_0_z_H, level_0_z_W, level_0_x_H, level_0_x_W, level_0_self_attention_q_positional_encoding, level_0_self_attention_k_positional_encoding)
        level_0_z, level_0_x = merged_0[:, :level_0_z_size, :], merged_0[:, level_0_z_size:, :]

        level_1_z_size = level_1_z_H * level_1_z_W
        level_1_x_size = level_1_x_H * level_1_x_W
        assert level_1_z_size == level_1_z.shape[1]
        assert level_1_x_size == level_1_x.shape[1]
        merged_1 = torch.cat((level_1_z, level_1_x), dim=1)
        for level_1_self_attention, level_1_self_attention_q_positional_encoding, level_1_self_attention_k_positional_encoding in zip(
                self.level_1_self_attentions, level_1_self_attention_q_positional_encodings, level_1_self_attention_k_positional_encodings):
            merged_1 = level_1_self_attention(merged_1, level_1_z_H, level_1_z_W, level_1_x_H, level_1_x_W, level_1_self_attention_q_positional_encoding, level_1_self_attention_k_positional_encoding)
        level_1_z, level_1_x = merged_1[:, :level_1_z_size, :], merged_1[:, level_1_z_size:, :]

        level_0_z, level_1_z = self.level_0_1_z_cross_attention(level_0_z, level_1_z, level_0_z_H, level_0_z_W, level_1_z_H, level_1_z_W, level_0_z_cross_attention_positional_encoding, level_1_z_cross_attention_positional_encoding)
        level_0_x, level_1_x = self.level_0_1_x_cross_attention(level_0_x, level_1_x, level_0_x_H, level_0_x_W, level_1_x_H, level_1_x_W, level_0_x_cross_attention_positional_encoding, level_1_x_cross_attention_positional_encoding)

        return level_0_z, level_0_x, level_1_z, level_1_x
