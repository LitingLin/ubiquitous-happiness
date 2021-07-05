import torch

from data.tracking.processor.siamfc_curation import do_SiamFC_curation
from .pipeline import build_TransT_image_augmentation_transformer


class TransTStage2DataProcessor:
    def __init__(self, template_size, search_size, color_jitter, interpolation_mode, do_imagenet_normalization):
        self.template_size = template_size
        self.search_size = search_size
        self.interpolation_mode = interpolation_mode
        self.transform = build_TransT_image_augmentation_transformer(color_jitter, do_imagenet_normalization)
        self.do_imagenet_normalization = do_imagenet_normalization

    def __call__(self, samples, miscellanies_host, miscellanies_device):
        z_image_batch_list, x_image_batch_list = samples
        # z_curation_parameters, x_curation_parameters, z_indices_in_collated_tensors, x_indices_in_collated_tensors = miscellanies_device
        # z_indices_in_collated_tensors, x_indices_in_collated_tensors = miscellanies_device

        z_indices_of_collated_tensors = miscellanies_host['z_index_of_collated_tensors']
        x_indices_of_collated_tensors = miscellanies_host['x_index_of_collated_tensors']
        z_curation_parameters = miscellanies_host['z_curation_parameters']
        x_curation_parameters = miscellanies_host['x_curation_parameters']
        z_indices_in_collated_tensors = miscellanies_host['z_index_in_collated_tensors']
        x_indices_in_collated_tensors = miscellanies_host['x_index_in_collated_tensors']

        z_batch = torch.empty([len(z_indices_of_collated_tensors), 3, *self.template_size], device=z_image_batch_list[0].device)
        x_batch = torch.empty([len(x_indices_of_collated_tensors), 3, *self.search_size], device=x_image_batch_list[0].device)
        for index, (z_index_of_collated_tensors, z_index_in_collated_tensors, z_context_element) in enumerate(zip(z_indices_of_collated_tensors, z_indices_in_collated_tensors, z_curation_parameters)):
            z_image = z_image_batch_list[z_index_of_collated_tensors][z_index_in_collated_tensors]
            z_batch[index, ...], _ = do_SiamFC_curation(z_image, self.template_size, z_context_element, self.interpolation_mode)
        if not self.do_imagenet_normalization:
            z_batch /= 255.0
        z_batch = self.transform(z_batch)
        if not self.do_imagenet_normalization:
            z_batch *= 255.0
        for index, (x_index_of_collated_tensors, x_index_in_collated_tensors, x_context_element) in enumerate(zip(x_indices_of_collated_tensors, x_indices_in_collated_tensors, x_curation_parameters)):
            x_image = x_image_batch_list[x_index_of_collated_tensors][x_index_in_collated_tensors]
            x_batch[index, ...], _ = do_SiamFC_curation(x_image, self.search_size, x_context_element, self.interpolation_mode)
        if not self.do_imagenet_normalization:
            x_batch /= 255.0
        x_batch = self.transform(x_batch)
        if not self.do_imagenet_normalization:
            x_batch *= 255.0
        return z_batch, x_batch
