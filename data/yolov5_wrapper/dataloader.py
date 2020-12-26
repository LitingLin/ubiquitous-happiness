from Utils.yolov5.torch_utils import torch_distributed_zero_first
from .wrapper import YoloV5Dataset
from .collate_fn import collate_fn
import torch.utils.data


def create_dataloader(dataset, imgsz, batch_size, stride, hyp=None, augment=False, pad=0.0, rect=False,
                      rank=-1, workers=8):
    with torch_distributed_zero_first(rank):
        dataset = YoloV5Dataset(dataset, imgsz, augment, hyp, rect, batch_size, int(stride), pad)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=workers, sampler=sampler, pin_memory=True, collate_fn=collate_fn)
    return dataloader, dataset
