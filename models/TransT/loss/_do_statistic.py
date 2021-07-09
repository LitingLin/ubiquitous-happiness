from miscellanies.torch.distributed.reduce_dict import reduce_dict


def do_statistic(stats: dict, weight_dict: dict):
    stats = {k: v.detach() for k, v in stats.items()}
    loss_dict_reduced = reduce_dict(stats)
    loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                  for k, v in loss_dict_reduced.items()}
    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                for k, v in loss_dict_reduced.items() if k in weight_dict}
    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

    loss_value = losses_reduced_scaled.item()

    loss_dict_reduced_scaled.update(loss_dict_reduced_unscaled)
    return loss_value, loss_dict_reduced_scaled
