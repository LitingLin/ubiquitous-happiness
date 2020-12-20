from Utils.yolov5_torch_utils import torch_distributed_zero_first
from .wrapper import YoloV5Dataset
from .collate_fn import collate_fn
from Dataset.Builder.builder import build_datasets
from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import torch.utils.data


def create_dataloader(path, imgsz, batch_size, stride, hyp=None, augment=False, pad=0.0, rect=False,
                      rank=-1, workers=8):
    dataset = build_datasets(path)
    assert len(dataset) == 1 and isinstance(dataset[0], DetectionDataset_MemoryMapped)
    dataset = dataset[0]

    with torch_distributed_zero_first(rank):
        dataset = YoloV5Dataset(dataset, imgsz, augment, hyp, rect, batch_size, int(stride), pad)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=workers, sampler=sampler, pin_memory=True, collate_fn=collate_fn)
    return dataloader, dataset
