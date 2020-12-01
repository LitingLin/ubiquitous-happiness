import torch
from typing import Iterable
import detr.util.misc as utils


@torch.no_grad()
def evaluate(actor, data_loader: Iterable, device):
    actor.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for exemplar, samples, targets in metric_logger.log_every(data_loader, 10, header):
        exemplar = exemplar.to(device)
        samples = samples.to(device)
        for t in targets:
            t['bbox'] = t['bbox'].to(device)

        stats = actor.forward(exemplar, samples, targets)

        metric_logger.update(**stats)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
