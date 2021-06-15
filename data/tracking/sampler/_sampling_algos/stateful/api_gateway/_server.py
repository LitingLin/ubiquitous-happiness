import numpy as np


class ApiGatewayRandomSamplerServer:
    def __init__(self, datasets, seed):
        self.rng_engine = np.random.Generator(np.random.PCG64(seed))
        seed += 1
        self.datasets = []
        self.dataset_ids = []
        for dataset in datasets:
            assert len(dataset) > 0
            indices = np.arange(len(dataset))
            shuffle_seed = seed
            rng_engine = np.random.Generator(np.random.PCG64(shuffle_seed))
            rng_engine.shuffle(indices)
            seed += 1
            self.datasets.append([len(dataset), indices, shuffle_seed, 0, 0])
            self.dataset_ids.append(dataset.get_unique_id())

    def __call__(self, command, response):
        if command[0] == 'get_status':
            str_ = ''
            for dataset_id, dataset_state in zip(self.dataset_ids, self.datasets):
                str_ += f'{dataset_id}({dataset_state[0]}): {dataset_state[3]} times + {dataset_state[4]}\n'
            response.set_body(str_)
        elif command[0] == 'get_next':
            index_of_dataset = self.rng_engine.integers(len(self.datasets))
            dataset_state = self.datasets[index_of_dataset]
            dataset_length, dataset_indices, _, _, _ = dataset_state
            response.set_body((index_of_dataset, dataset_indices[dataset_state[4]]))
            response.commit()
            dataset_state[4] += 1
            if dataset_state[4] == dataset_length:
                dataset_state[2] += 1
                seed = dataset_state[2]
                rng_engine = np.random.Generator(np.random.PCG64(seed))
                rng_engine.shuffle(dataset_indices)
                dataset_state[3] += 1
                dataset_state[4] = 0
        elif command[0] == 'get_state':
            response.set_body(self.get_state())
        elif command[0] == 'set_state':
            self.set_state(command[1])
        else:
            raise Exception(f'Unknown command recerived {command}')

    def get_state(self):
        return self.rng_engine.__getstate__(), self.datasets

    def set_state(self, state):
        self.rng_engine.__setstate__(state[0])
        self.datasets = state[1]
