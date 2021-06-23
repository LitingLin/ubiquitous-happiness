import torch.nn as nn


class SiamFCCriterion(nn.Module):
    def __init__(self, criterion):
        super(SiamFCCriterion, self).__init__()
        self.criterion = criterion

    def forward(self, predicted, target):
        loss = self.criterion(predicted, target)
        return loss, loss.item(), {}
