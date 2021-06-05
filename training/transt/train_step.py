from Miscellaneous.torch.metric_logger import MetricLogger
from Miscellaneous.torch.smoothed_value import SmoothedValue
from typing import Iterable


def train_one_epoch(runner, data_loader: Iterable, epoch: int, max_norm: float = 0):
    runner.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        forward_stats = runner.forward(*data)
        backward_stats = runner.backward(max_norm)

        metric_logger.update(**forward_stats)
        metric_logger.update(**backward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
