from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from PIL import Image

class ColorNormDataset(Dataset):
    def __init__(self, txt_path='filepath'):
        self.images = list(open(txt_path))

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img.strip())
        img = img.convert('RGB')
        sample = transforms.ToTensor()(img)
        sample_jitter = sample.clone()
        r, b = random.random()/2, random.random()/2
        sample_jitter[0] += r - 0.25
        sample_jitter[2] += b - 0.25
        return sample, sample_jitter

    def __len__(self):
        return len(self.images)


class ColorNormDatasetInf(Dataset):
    def __init__(self, txt_path='filepath'):
        self.images = list(open(txt_path))

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img.strip())
        img = img.convert('RGB')
        sample = transforms.ToTensor()(img)
        return sample

    def __len__(self):
        return len(self.images)
