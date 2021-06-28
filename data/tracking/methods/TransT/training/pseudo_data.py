import torch


class TransTPseudoDataGenerator:
    def __init__(self, template_size, search_size, device):
        self.template_size = template_size
        self.search_size = search_size
        self.device = device

    def get(self, batch):
        return torch.empty((batch, 3, *self.template_size), device=self.device), \
            torch.empty((batch, 3, *self.search_size), device=self.device)
