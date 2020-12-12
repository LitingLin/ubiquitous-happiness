import torch


class DETRSiamFCWrapper:
    def __init__(self, siamfc, position_encoder):
        self.siamfc = siamfc
        self.position_encoder = position_encoder

    def _generate_mask_position_enc(self, response):
        b, h, w = response.shape
        mask = torch.ones((b, h, w), dtype=torch.bool, device=response.device)
        position = self.position_encoder(response)
        return mask, position

    def forward(self, z, x):
        response = self.siamfc(z, x)
        if isinstance(response, list):
            mask = []
            pos_enc = []
            for cur_response in response:
                c_mask, c_pos_enc = self._generate_mask_position_enc(cur_response)
                mask.append(c_mask)
                pos_enc.append(c_pos_enc)
        else:
            mask, pos_enc = self._generate_mask_position_enc(response)
        return response, mask, pos_enc
