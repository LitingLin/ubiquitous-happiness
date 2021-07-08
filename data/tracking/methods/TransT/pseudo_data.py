import torch


class TransTPseudoDataGenerator:
    def __init__(self, template_size, search_size, device: torch.device):
        self.template_size = template_size
        self.search_size = search_size
        self.device = device

    def get_train(self, batch):
        return torch.empty((batch, 3, *self.template_size), device=self.device), \
            torch.empty((batch, 3, *self.search_size), device=self.device)

    def is_cuda(self):
        return 'cuda' in self.device.type

    def get_device(self):
        return self.device


def build_pseudo_data_generator(args, network_config: dict):
    device = torch.device(args.device)
    network_data_config = network_config['data']
    return TransTPseudoDataGenerator(network_data_config['template_size'], network_data_config['search_size'], device)
