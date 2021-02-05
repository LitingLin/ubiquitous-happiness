import torch


def _get_tensor_list_max_hw(tensors):
    c, max_h, max_w = tensors[0].shape
    for tensor in tensors[1:]:
        t_c, t_h, t_w = tensor.shape
        assert c == t_c
        if max_h < t_h:
            max_h = t_h
        if max_w < t_w:
            max_w = t_w
    return c, max_h, max_w


def collate_different_size_4D_tensors_and_generate_masks(tensors):
    c, h, w = _get_tensor_list_max_hw(tensors)
    b = len(tensors)
    collated_tensors = torch.zeros((b, c, h, w), dtype=tensors[0].dtype, device=tensors[0].device)
    masks = torch.ones((b, h, w), dtype=torch.bool, device=tensors[0].device)
    for tensor, collated_tensor, mask in zip(tensors, collated_tensors, masks):
        collated_tensor[:, : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
        mask[: tensor.shape[1], :tensor.shape[2]] = False

    return collated_tensors, masks
