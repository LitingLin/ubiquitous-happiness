import torch
import gc


def _default_tensor_list_fn(data):
    tensor_list = []
    if isinstance(data, (list, tuple)):
        for i in data:
            tensor_list.extend(_default_tensor_list_fn(i))
    elif isinstance(data, dict):
        for v in data.values():
            tensor_list.extend(_default_tensor_list_fn(v))
    elif isinstance(data, torch.Tensor):
        tensor_list.append(data)
    return tensor_list


def _default_regroup_fn_(data, device_tensors: list):
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = device_tensors.pop(0)
            elif isinstance(data[i], (list, tuple, dict)):
                data[i] = _default_regroup_fn_(data[i], device_tensors)
    elif isinstance(data, dict):
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = device_tensors.pop(0)
            elif isinstance(data[k], (list, tuple, dict)):
                data[k] = _default_regroup_fn_(data[k], device_tensors)
    elif isinstance(data, torch.Tensor):
        return device_tensors.pop(0)

    return data


class DefaultTensorFilter:
    @staticmethod
    def get_tensor_list(data):
        return _default_tensor_list_fn(data)

    @staticmethod
    def regroup(data, device_tensors):
        return _default_regroup_fn_(data, device_tensors)


class TensorFilteringByIndices:
    def __init__(self, indices):
        self.indices = indices

    def get_tensor_list(self, data):
        split_points = []
        device_tensor_list = []
        for index in self.indices:
            datum = data[index]
            if datum is not None:
                device_tensors = _default_tensor_list_fn(datum)
                split_points.append(len(device_tensors))
                device_tensor_list.extend(device_tensors)
        return device_tensor_list

    def regroup(self, data, device_tensors: list):
        collated = []
        for index, datum in enumerate(data):
            if index in self.indices and datum is not None:
                datum = _default_regroup_fn_(datum, device_tensors)
            collated.append(datum)
        return collated


class CUDAPrefetcher:
    def __init__(self, iterator, device=None, tensor_filter=None):
        if tensor_filter is None:
            tensor_filter = DefaultTensorFilter
        self.iterator = iterator
        if device is None:
            device = torch.device('cuda')
        self.device = device
        self.tensor_filter = tensor_filter
        self.tensor_list = None

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.iter = iter(self.iterator)
        self.preload()
        assert self.tensor_list is not None, "empty iterator"
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
        data = self.tensor_filter.regroup(data, tensor_list)
        assert len(tensor_list) == 0
        return data

    def preload(self):
        try:
            self.data = next(self.iter)
        except StopIteration:
            self.data = None
            self.tensor_list = None
            return

        self.tensor_list = self.tensor_filter.get_tensor_list(self.data)

        with torch.cuda.stream(self.stream):
            for i in range(len(self.tensor_list)):
                self.tensor_list[i] = self.tensor_list[i].to(self.device, non_blocking=True)
