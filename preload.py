import os

import torch
import hydra
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from data.dataset import FineImageFolder

from utils.func import *
from modules.builder import build_frozen_encoder


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    if os.path.exists(cfg.dataset.preload_path):
        print_msg('Preload path {} exists.'.format(cfg.dataset.preload_path), warning=True)
        print('WARNING: Please update the "preload_path" in "/configs/dataset" '
              'or add an argument "++dataset.preload_path=new_path" in the command.')
        return

    print('Preloading {} dataset...'.format(cfg.dataset.name))
    frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)
    dataset = generate_dataset(cfg)
    preload_dataset(cfg, dataset, frozen_encoder)
    print('Preloading done.')


def preload_dataset(cfg, dataset, frozen_encoder):
    train_dataset, test_dataset, val_dataset = dataset
    print('Preloading train dataset...')
    preload(cfg, train_dataset, frozen_encoder, cfg.dataset.preload_path)
    print('Preloading test dataset...')
    preload(cfg, test_dataset, frozen_encoder, cfg.dataset.preload_path)
    print('Preloading val dataset...')
    preload(cfg, val_dataset, frozen_encoder, cfg.dataset.preload_path)


def generate_dataset(cfg):
    data_path = cfg.dataset.data_path

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    preprocess = transforms.Compose([
        transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.mean, cfg.dataset.std)
    ])

    train_dataset = FineImageFolder(train_path, preprocess)
    test_dataset = FineImageFolder(test_path, preprocess)
    val_dataset = FineImageFolder(val_path, preprocess)

    return train_dataset, test_dataset, val_dataset


def preload(cfg, dataset, frozen_encoder, preload_path='./preload_data'):
    os.makedirs(preload_path, exist_ok=True)

    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    for img_paths, X in tqdm(loader):
        img_names = [Path(path).stem for path in img_paths]
        X = X.to(cfg.base.device)
        with torch.no_grad():
            _, key_states, value_states = frozen_encoder(X, interpolate_pos_encoding=True)
            key_states = key_states.cpu()
            value_states = value_states.cpu()

            for i in range(len(img_names)):
                states = {
                    'key_states': key_states[:, i].contiguous(),
                    'value_states': value_states[:, i].contiguous()
                }
                save_path = os.path.join(preload_path, img_names[i] + '.safetensors')
                save_file(states, save_path)


if __name__ == '__main__':
    main()