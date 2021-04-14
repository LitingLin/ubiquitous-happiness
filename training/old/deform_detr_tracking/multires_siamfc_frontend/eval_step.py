import torch
import Utils.detr_misc as utils


@torch.no_grad()
def evaluate(actor, data_loader, device):
    actor.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for z, x, x_bbox in metric_logger.log_every(data_loader, 10, header):
        z = z.to(device)
        x = x.to(device)
        x_bbox = x_bbox.to(device)

        with torch.no_grad():
            forward_stats = actor.forward(z, x, x_bbox)

        metric_logger.update(**forward_stats)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
