import torch
from Miscellaneous.torch.metric_logger import MetricLogger


@torch.no_grad()
def evaluate(runner, data_loader):
    runner.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        with torch.no_grad():
            forward_stats = runner.forward(samples, targets)

        metric_logger.update(**forward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
