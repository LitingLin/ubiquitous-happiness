def batchify(tensor):
    return tensor.unsqueeze(0)


def unbatchify(tensor):
    return tensor.squeeze()
