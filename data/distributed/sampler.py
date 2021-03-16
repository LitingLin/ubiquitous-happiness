import copy


class SequentialDatasetSamplerSynchronizationServiceServer:
    def __init__(self, datasets):
        self.dataset_indices = [(index_of_dataset, list(range(len(dataset)))) for index_of_dataset, dataset in enumerate(datasets)]
        self.indices = None

    def __call__(self, command, response):
        if command == 'reset':
            self.indices = copy.deepcopy(self.dataset_indices)
        elif command == 'next':
            if len(self.indices) == 0 or self.indices is None:
                response.set_body('empty')
            else:
                index_of_dataset, index_of_sequences = self.indices[0]
                index_of_sequence = index_of_sequences.pop(0)
                if len(index_of_sequences) == 0:
                    self.indices.pop(0)
                response.set_body((index_of_dataset, index_of_sequence))
        else:
            response.set_status_code(400)

from .notication_based_sampler import Client
class SequentialDatasetSamplerSynchronizationServiceClient:
    def __init__(self, server_address):
        self.client = Client(server_address)

    def reset(self):
        self.client('reset')

    def next(self):
        response = self.client('next')
        if response == 'empty':
            return False, None
        else:
            return True, response


class SequentialDatasetSamplerIterator:
    def __init__(self, synchronization_service_client):
        self.synchronization_service_client = synchronization_service_client

    def __next__(self):
        has_next, indices = self.synchronization_service_client.next()
        if not has_next:
            raise StopIteration
        return indices
    #return dataset sequence, index of frame



class DistributedPreemptiveSchedulingSingleObjectTrackingDatasetSampler:
    def __init__(self, datasets, remote_api_server_address, batch_size, num_replicas, rank):

        self.datasets = datasets
        self.synchronization_service = SequentialDatasetSamplerSynchronizationServiceClient(remote_api_server_address)

    def __iter__(self):
        pass

    def __len__(self):
        # estimated
        pass

    def reset(self):
        self.synchronization_service.reset()


class SequentialDataset:
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass


