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
    def __init__(self, root, lpm_transform=None, side_transform=None, loader=pil_loader):
        super(AsymetricImageFolder, self).__init__(root, loader=loader)
        self.lpm_transform = lpm_transform
        self.side_transform = side_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        lpm_sample = self.lpm_transform(sample)
        side_sample = self.side_transform(sample)

        return lpm_sample, side_sample, target


class PreloadImageFolder(datasets.ImageFolder):
    def __init__(self, root, preload_path, side_transform=None, loader=pil_loader):
        super(PreloadImageFolder, self).__init__(root, loader=loader)
        self.preload_path = preload_path
        self.side_transform = side_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        side_sample = self.side_transform(sample)
        key_states, value_states = self.preload(path)
        return side_sample, key_states, value_states, target

    def preload(self, path):
        states_path = os.path.join(self.preload_path, Path(path).stem + '.safetensors')
        states = load_file(states_path)
        key_states, value_states = states['key_states'], states['value_states']
        return key_states, value_states


class FineImageFolder(datasets.ImageFolder):
    def __init__(self, root, lpm_transform=None, loader=pil_loader):
        super(FineImageFolder, self).__init__(root, loader=loader)
        self.lpm_features = {}
        self.lpm_transform = lpm_transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        lpm_sample = self.lpm_transform(sample)
        return path, lpm_sample
