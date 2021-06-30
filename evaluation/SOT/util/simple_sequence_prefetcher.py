from miscellanies.simple_prefetcher import SimplePrefetcher
import torchvision.io


class _Sequence_Data_Getter:
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, index: int):
        frame = self.sequence[index]
        return torchvision.io.read_image(frame.get_image_path(),
                                         torchvision.io.image.ImageReadMode.RGB), \
               frame.get_bounding_box(), frame.get_bounding_box_validity_flag()

    def __len__(self):
        return len(self.sequence)


def get_simple_sequence_data_prefetcher(sequence):
    sequence_data_getter = _Sequence_Data_Getter(sequence)
    sequence_data_getter = SimplePrefetcher(sequence_data_getter)
    return sequence_data_getter
