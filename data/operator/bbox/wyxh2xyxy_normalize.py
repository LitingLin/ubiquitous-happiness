import torch


def bbox_wyxh2xyxy_normalize(bbox, image_size):
    bbox = [bbox[0] + bbox[2] / 2,
                           bbox[1] + bbox[3] / 2,
                           bbox[2], bbox[3]]

    w, h = image_size
    bbox = [bbox[0] / w, bbox[1] / h,
                           bbox[2] / w, bbox[3] / h]
    return torch.tensor(bbox, dtype=torch.float)
