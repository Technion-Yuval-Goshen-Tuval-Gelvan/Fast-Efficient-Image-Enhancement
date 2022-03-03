from torch.utils.data import Dataset
import glob
from torchvision.io import read_image
import os


class Div2k(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.data = glob.glob(os.path.join(folder, '*'))
        self.transform = transform

    def __getitem__(self, index):
        im = read_image(self.data[index]) / 255.0
        if self.transform:
            im = self.transform(im)
        return im

    def __len__(self):
        return len(self.data)
