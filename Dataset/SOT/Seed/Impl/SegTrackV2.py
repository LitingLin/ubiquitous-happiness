from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor
import os


def construct_SegTrackV2(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    constructor.set_total_number_of_sequences()

    sequences = os.listdir(root_path)
    for sequence in sequences:
