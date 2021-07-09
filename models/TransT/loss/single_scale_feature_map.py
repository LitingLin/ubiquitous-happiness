import torch.nn as nn
from ._criterion import criterion
from ._do_statistic import do_statistic


class SingleScaleFeatureMapCriterion(nn.Module):
    def __init__(self, loss_dict: nn.ModuleDict, loss_target_dict: dict, weight_dict: dict):
        super().__init__()
        self.loss_dict = loss_dict
        self.loss_target_dict = loss_target_dict
        self.weight_dict = weight_dict

    def forward(self, predicted, label):
        loss_dict = criterion(predicted, label, self.loss_dict, self.loss_target_dict)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
        return (losses, *do_statistic(loss_dict, self.weight_dict))
