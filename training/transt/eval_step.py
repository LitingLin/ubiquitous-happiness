import torch
from Miscellaneous.torch.metric_logger import MetricLogger
import gc


@torch.no_grad()
def evaluate(runner, logger, data_loader, logging_interval):
    runner.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    gc.collect()
    for data in metric_logger.log_every(data_loader, logging_interval, header):
        with torch.no_grad():
            forward_stats = runner.forward(*data)

        metric_logger.update(**forward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logger.log_test(runner.get_epoch(), stats)
    return stats
