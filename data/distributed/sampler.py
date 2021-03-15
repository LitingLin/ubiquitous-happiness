class SequentialDatasetSamplerAPIServer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.indices = None

    def __call__(self, command, response):
        if command == 'reset':
            self.indices = list(range(len(self.datasets)))
        elif command == 'new':
            response.set_body(self.indices.pop(0))
        else:
            response.set_status_code(400)

from .notication_based_sampler import Client
class SequentialDatasetSamplerSynchronizationServiceClient:
    def __init__(self, server_address):
        self.client = Client(server_address)



class SequentialDatasetSamplerIterator:
    def __init__(self):
        pass



class SequentialDatasetSampler:
    def __init__(self, datasets, remote_api_server_address):
        self.datasets = datasets
        self.remote_api_server_address = remote_api_server_address

    def __len__(self):
        pass

    def __iter__(self):
        pass