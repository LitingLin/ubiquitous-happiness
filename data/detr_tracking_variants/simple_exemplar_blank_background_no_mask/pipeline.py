import numpy as np
from ..simple.pipeline import SimpleDataPipelineOrganizer


class SimpleDataPipelineOrganizer_NoZMask(SimpleDataPipelineOrganizer):
    def __call__(self, z_output, x_output):
        z, _ = z_output
        z_h, z_w = z.shape[0: 2]
        z = SimpleDataPipelineOrganizer._common_image_post_process(z)
        z_mask = np.zeros((z_h, z_w), dtype=np.bool)
        x, x_bbox = x_output
        x, x_bbox = SimpleDataPipelineOrganizer._common_post_process(x, x_bbox)
        return z, z_mask, x, x_bbox
