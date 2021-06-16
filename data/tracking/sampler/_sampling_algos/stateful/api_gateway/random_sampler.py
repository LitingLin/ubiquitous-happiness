from Miscellaneous.simple_api_gateway import ServerLauncher, Client
from Miscellaneous.torch.distributed import is_main_process
from data.tracking.sampler._sampling_algos.stateful.api_gateway._server import ApiGatewayRandomSamplerServer


class ApiGatewayRandomSampler:
    def __init__(self, datasets, datasets_sampling_probability, socket_address, seed: int):
        self.datasets = datasets
        if is_main_process():
            self.server_callback = ApiGatewayRandomSamplerServer(datasets, datasets_sampling_probability, seed)
            self.server = ServerLauncher(socket_address, self.server_callback)
        self.client = Client(socket_address)

    def __del__(self):
        if is_main_process():
            self.server.stop()

    def launch(self):
        if is_main_process():
            self.server.launch()

    def stop(self):
        if is_main_process():
            self.server_callback.set_state(self.client('get_state', ))
            self.server.stop()

    def get_state(self):
        assert is_main_process()

        if self.server.is_launched():
            state = self.client('get_state', )
            self.server_callback.set_state(state)
            return state
        else:
            return self.server_callback.get_state()

    def set_state(self, state):
        assert is_main_process()

        if self.server.is_launched():
            self.client('set_state', state)
            self.server_callback.set_state(state)
        else:
            self.server_callback.set_state(state)

    def get_next(self):
        index_of_dataset, index_of_sequence = self.client('get_next', )
        return index_of_dataset, index_of_sequence

    def get_status(self):
        return self.client('get_status', )
