import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
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
        image = Image.open(frame.getImagePath()).convert('RGB')
        w, h = frame.getImageSize()

