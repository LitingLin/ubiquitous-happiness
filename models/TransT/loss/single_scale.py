import torch.nn as nn


def _compute_loss(pred, label, context, input_hook, loss_functions, loss_data_adaptors, loss_reduce_function):
    if input_hook is not None:
        pred, label = input_hook(pred, label, context)
    losses = []
    for loss_fn, loss_data_adaptor in zip(loss_functions, loss_data_adaptors):
        shall_skip, loss_fn_inputs = loss_data_adaptor(pred, label, context)
        if shall_skip:
            loss = loss_fn_inputs
        else:
            if isinstance(loss_fn_inputs, (list, tuple)):
                loss = loss_fn(*loss_fn_inputs)
            elif isinstance(loss_fn_inputs, dict):
                loss = loss_fn(**loss_fn_inputs)
            else:
                loss = loss_fn(loss_fn_inputs)
            loss = loss_reduce_function(loss, pred, label, context)
        losses.append(loss)
    return losses


class SingleScaleCriterion(nn.Module):
    def __init__(self, global_data_filter, loss_modules):
        super(SingleScaleCriterion, self).__init__()
        self.global_data_filter = global_data_filter
        for module_name, module_input_hook, module_loss_functions, module_loss_reduce_function in loss_modules:
            self.__setattr__(module_name, nn.ModuleList(module_loss_functions))
        self.loss_modules = loss_modules

    def forward(self, pred, label):
        context = {}
        if self.global_data_filter is not None:
            pred, label = self.global_data_filter(pred, label, context)
        losses = []
        for module_name, module_data_filter, _, loss_data_adaptors, module_loss_reduce_function in self.loss_modules:
            module_loss_functions = self.__getattr__(module_name)
            losses.extend(_compute_loss(pred, label, context, module_data_filter, module_loss_functions, loss_data_adaptors, module_loss_reduce_function))
        return losses
