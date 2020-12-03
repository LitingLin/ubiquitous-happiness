from torch.utils.data.dataset import Dataset
from ._common import _detr_processing
import random


class DETRSOTDataset(Dataset):
    def __init__(self, dataset, transforms=None, samples_per_vid=1):
        self.dataset = dataset
        self.samples_per_vid = samples_per_vid
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset) * self.samples_per_vid

    def __getitem__(self, index: int):
        index_of_sequence = index // self.samples_per_vid
        sequence = self.dataset[index_of_sequence]
        index_of_frame = random.randint(0, len(sequence) - 1)
        frame = sequence[index_of_frame]
        image_path = frame.getImagePath()
        image_size = frame.getImageSize()

        box = [frame.getBoundingBox()]
        class_ = [sequence.getCategoryName()]
        return _detr_processing(image_path, image_size, None, box, class_, self.transforms)
