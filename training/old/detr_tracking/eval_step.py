import torch
import Utils.detr_misc as utils


@torch.no_grad()
def evaluate(actor, data_loader):
    actor.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        with torch.no_grad():
            forward_stats = actor.forward(samples, targets)

        metric_logger.update(**forward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
