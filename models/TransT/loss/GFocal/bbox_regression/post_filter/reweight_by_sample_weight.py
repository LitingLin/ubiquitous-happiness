from miscellanies.torch.distributed.reduce_mean import reduce_mean_


def reweight_by_sample_weight(losses, predicted, label, context):
    weight = context['sample_weight']

    weight = weight.sum()
    reduce_mean_(weight)
    return [loss / weight for loss in losses]


def build_data_filter(_):
    return reweight_by_sample_weight
