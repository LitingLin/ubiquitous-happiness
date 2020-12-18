import torch
import Utils.detr_misc as utils
from typing import Iterable


def train_one_epoch(actor,
                    data_loader: Iterable,
                    device: torch.device, epoch: int, max_norm: float = 0):
    actor.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for z, x, x_bbox in metric_logger.log_every(data_loader, print_freq, header):
        z = z.to(device)
        x = x.to(device)
        x_bbox = x_bbox.to(device)

        forward_stats = actor.forward((z, x), x_bbox)
        backward_stats = actor.backward(max_norm)

        metric_logger.update(**forward_stats)
        metric_logger.update(**backward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
