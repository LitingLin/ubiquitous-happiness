import torch.nn as nn


def _compute_loss(pred, label, input_hook, loss_functions, loss_data_adaptors, loss_reduce_function):
    if input_hook is not None:
        pred, label = input_hook(pred, label)
    losses = []
    for loss_fn, loss_data_adaptor in zip(loss_functions, loss_data_adaptors):
        loss = loss_fn(*loss_data_adaptor(pred, label))
        loss = loss_reduce_function(loss)
        losses.append(loss)
    return losses


class SingleScaleCriterion(nn.Module):
    def __init__(self, loss_modules):
        super(SingleScaleCriterion, self).__init__()
        for module_name, module_input_hook, module_loss_functions, module_loss_reduce_function in loss_modules:
            self.__setattr__(module_name, nn.ModuleList(module_loss_functions))
        self.loss_modules = loss_modules

    def forward(self, pred, label):
        losses = []
        for module_name, module_data_filter, _, loss_data_adaptors, module_loss_reduce_function in self.loss_modules:
            module_loss_functions = self.__getattr__(module_name)
            losses.extend(_compute_loss(pred, label, module_data_filter, module_loss_functions, loss_data_adaptors, module_loss_reduce_function))
        return losses
