def collate(lists):
    return tuple(map(tuple, zip(*lists)))
