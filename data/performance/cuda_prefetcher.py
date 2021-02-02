import torch
import gc


def default_tensor_list_fn(data):
    tensor_list = []
    if isinstance(data, (list, tuple)):
        for i in data:
            tensor_list.extend(default_tensor_list_fn(i))
    elif isinstance(data, dict):
        for v in data.values():
            tensor_list.extend(default_tensor_list_fn(v))
    elif isinstance(data, torch.Tensor):
        tensor_list.append(data)
    return tensor_list


def default_regroup_fn(data, cuda_tensors: list):
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = cuda_tensors.pop(0)
            elif isinstance(data[i], (list, tuple, dict)):
                data[i] = default_regroup_fn(data[i], cuda_tensors)
    elif isinstance(data, dict):
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = cuda_tensors.pop(0)
            elif isinstance(data[k], (list, tuple, dict)):
                data[k] = default_regroup_fn(data[k], cuda_tensors)
    elif isinstance(data, torch.Tensor):
        return cuda_tensors.pop(0)

    return data


class CUDAPrefetcher:
    def __init__(self, data_loader, device=None, tensor_list_fn=default_tensor_list_fn, regroup_fn=default_regroup_fn):
        self.data_loader = data_loader
        if device is None:
            device = torch.device('cuda')
        self.device = device

        self.tensor_list_fn = tensor_list_fn
        self.regroup_fn = regroup_fn

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.iter = iter(self.data_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        data = self.data
        tensor_list = self.tensor_list

        if data is None:
            if hasattr(self, 'stream'):
                del self.stream
            gc.collect()
            raise StopIteration

        for tensor in tensor_list:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        data = self.regroup_fn(data, tensor_list)
        assert len(tensor_list) == 0
        return data

    def preload(self):
        try:
            self.data = next(self.iter)
        except StopIteration:
            self.data = None
            return

        self.tensor_list = self.tensor_list_fn(self.data)

        with torch.cuda.stream(self.stream):
            for i in range(len(self.tensor_list)):
                self.tensor_list[i] = self.tensor_list[i].to(self.device, non_blocking=True)
