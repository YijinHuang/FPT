import os

from torchvision import datasets

from .transforms import data_transforms, simple_transform
from .dataset import PreloadImageFolder, AsymetricImageFolder
from utils.func import mean_and_std, print_dataset_info


def generate_dataset(cfg):
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        mean, std = auto_statistics(
            cfg.base.data_path,
            cfg.data.fine_input_size,
            cfg.train.batch_size,
            cfg.train.num_workers
        )
        cfg.data.mean = mean
        cfg.data.std = std

    train_transform, test_transform = data_transforms(cfg)
    datasets = generate_dataset_from_folder(
        cfg,
        train_transform,
        test_transform,
    )

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path, input_size, batch_size, num_workers):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)
    train_path = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_dataset_from_folder(cfg, train_transform, test_transform):
    data_path = cfg.base.data_path
    preload_path = cfg.base.preload_path

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    if cfg.base.preload:
        train_dataset = PreloadImageFolder(train_path, preload_path, train_transform)
        test_dataset = PreloadImageFolder(test_path, preload_path, test_transform)
        val_dataset = PreloadImageFolder(val_path, preload_path, test_transform)
    else:
        train_fine_transform, train_coarse_transform = train_transform
        test_fine_transform, test_coarse_transform = test_transform

        train_dataset = AsymetricImageFolder(train_path, train_fine_transform, train_coarse_transform)
        test_dataset = AsymetricImageFolder(test_path, test_fine_transform, test_coarse_transform)
        val_dataset = AsymetricImageFolder(val_path, test_fine_transform, test_coarse_transform)

    dataset = train_dataset, test_dataset, val_dataset
    return dataset
