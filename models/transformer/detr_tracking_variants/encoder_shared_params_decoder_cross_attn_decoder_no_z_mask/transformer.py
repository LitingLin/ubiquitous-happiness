from torch import nn, Tensor
from models.transformer.detr_transformer import TransformerEncoderLayer, TransformerEncoder
from models.transformer.detr_tracking_variants.encoder_shared_params_decoder_cross_attn_decoder_no_z_mask.decoder import TransformerDecoder


def flatten_tensor_mask_position_encoding(tensor, mask, position):
    # flatten NxCxHxW to HWxNxC
    tensor = tensor.flatten(2).permute(2, 0, 1)
    mask = mask.flatten(1)
    position = position.flatten(2).permute(2, 0, 1)
    return tensor, mask, position


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, False)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)

        self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward, dropout, activation, num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_template(self, z, z_mask, z_pos):
        z, z_mask, z_pos = flatten_tensor_mask_position_encoding(z, z_mask, z_pos)
        z = self.encoder(z, src_key_padding_mask=z_mask, pos=z_pos)
        return z, z_mask, z_pos

    def forward_instance(self, z_encoded_flatten, z_mask_flatten, z_pos_flatten, x, x_mask, x_pos):
        x, x_mask, x_pos = flatten_tensor_mask_position_encoding(x, x_mask, x_pos)
        x = self.encoder(x, src_key_padding_mask=x_mask, pos=x_pos)
        bbox_embed = self.decoder(z_encoded_flatten, x, z_mask_flatten, x_mask, z_pos_flatten, x_pos)
        # to N x HW(z) x C
        return bbox_embed.transpose(0, 1)

    def forward(self, z, x, z_mask, x_mask, z_pos, x_pos):
        z, z_mask, z_pos = flatten_tensor_mask_position_encoding(z, z_mask, z_pos)
        x, x_mask, x_pos = flatten_tensor_mask_position_encoding(x, x_mask, x_pos)

        z = self.encoder(z, src_key_padding_mask=z_mask, pos=z_pos)
        x = self.encoder(x, src_key_padding_mask=x_mask, pos=x_pos)

        bbox_embed = self.decoder(z, x, z_mask, x_mask, z_pos, x_pos)
        # to N x HW(z) x C
        return bbox_embed.transpose(0, 1)
