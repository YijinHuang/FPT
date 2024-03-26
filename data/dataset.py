import os
from PIL import Image
from pathlib import Path
from torchvision import datasets
from safetensors.torch import load_file


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class AsymetricImageFolder(datasets.ImageFolder):
    def __init__(self, root, fine_transform=None, coarse_transform=None, loader=pil_loader):
        super(AsymetricImageFolder, self).__init__(root, loader=loader)
        self.fine_transform = fine_transform
        self.coarse_transform = coarse_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        fine_sample = self.fine_transform(sample)
        coarse_sample = self.coarse_transform(sample)

        return fine_sample, coarse_sample, target


class PreloadImageFolder(datasets.ImageFolder):
    def __init__(self, root, preload_path, coarse_transform=None, loader=pil_loader):
        super(PreloadImageFolder, self).__init__(root, loader=loader)
        self.preload_path = preload_path
        self.coarse_transform = coarse_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        coarse_sample = self.coarse_transform(sample)
        key_states, value_states = self.preload(path)
        return coarse_sample, key_states, value_states, target

    def preload(self, path):
        states_path = os.path.join(self.preload_path, Path(path).stem + '.safetensors')
        states = load_file(states_path)
        key_states, value_states = states['key_states'], states['value_states']
        return key_states, value_states


class FineImageFolder(datasets.ImageFolder):
    def __init__(self, root, fine_transform=None, loader=pil_loader):
        super(FineImageFolder, self).__init__(root, loader=loader)
        self.fine_features = {}
        self.fine_transform = fine_transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        fine_sample = self.fine_transform(sample)
        return path, fine_sample
