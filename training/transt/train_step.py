from Miscellaneous.torch.metric_logger import MetricLogger
from Miscellaneous.torch.smoothed_value import SmoothedValue
from typing import Iterable
import gc


def train_step(runner, logger, data_loader: Iterable, logging_interval):
    runner.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(runner.get_epoch())

    gc.collect()

    for data in metric_logger.log_every(data_loader, logging_interval, header):
        forward_stats = runner.forward(*data)
        backward_stats = runner.backward()

        logger.log_train(runner.get_epoch(), forward_stats, backward_stats)

        metric_logger.update(**forward_stats)
        metric_logger.update(**backward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
