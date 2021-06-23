import torch

from data.tracking.processor.siamfc_curation import do_SiamFC_curation
from .pipeline import build_TransT_image_augmentation_transformer


class TransTStage2DataProcessor:
    def __init__(self, template_size, search_size, color_jitter, interpolation_mode):
        self.template_size = template_size
        self.search_size = search_size
        self.interpolation_mode = interpolation_mode
        self.transform = build_TransT_image_augmentation_transformer(color_jitter)

    def __call__(self, samples, context):
        z, x = samples
        z_context, x_context = context

        z_batch = torch.empty([len(z), 3, *self.template_size], device=z[0].device)
        x_batch = torch.empty([len(x), 3, *self.search_size], device=x[0].device)
        for index, (z_element, z_context_element) in enumerate(zip(z, z_context)):
            z_batch[index, ...], _ = do_SiamFC_curation(z_element, self.template_size, z_context_element, self.interpolation_mode)
        z_batch = self.transform(z_batch)
        for index, (x_element, x_context_element) in enumerate(zip(x, x_context)):
            x_batch[index, ...], _ = do_SiamFC_curation(x_element, self.search_size, x_context_element, self.interpolation_mode)
        x_batch = self.transform(x_batch)
        # x, _ = do_SiamFC_curation(x, self.search_size, x_context)
        # x = self.transform(x)
        return z_batch, x_batch
