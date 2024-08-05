import torch
from torchvision import transforms
from packaging import version
from copy import deepcopy


def data_transforms(cfg):
    weak_data_aug = [
        "horizontal_flip",
        "color_distortion",
        "random_crop"
    ]
    aug_args = cfg.data_augmentation_args

    def get_operations(aug_args, input_size):
        operations = {
            'random_crop': random_apply(
                transforms.RandomResizedCrop(
                    size=(input_size, input_size),
                    scale=aug_args.random_crop.scale,
                    ratio=aug_args.random_crop.ratio
                ),
                p=aug_args.random_crop.prob
            ),
            'horizontal_flip': transforms.RandomHorizontalFlip(
                p=aug_args.horizontal_flip.prob
            ),
            'vertical_flip': transforms.RandomVerticalFlip(
                p=aug_args.vertical_flip.prob
            ),
            'color_distortion': random_apply(
                transforms.ColorJitter(
                    brightness=aug_args.color_distortion.brightness,
                    contrast=aug_args.color_distortion.contrast,
                    saturation=aug_args.color_distortion.saturation,
                    hue=aug_args.color_distortion.hue
                ),
                p=aug_args.color_distortion.prob
            ),
            'rotation': random_apply(
                transforms.RandomRotation(
                    degrees=aug_args.rotation.degrees,
                    fill=aug_args.value_fill
                ),
                p=aug_args.rotation.prob
            ),
            'translation': random_apply(
                transforms.RandomAffine(
                    degrees=0,
                    translate=aug_args.translation.range,
                    fill=aug_args.value_fill
                ),
                p=aug_args.translation.prob
            ),
            'grayscale': transforms.RandomGrayscale(
                p=aug_args.grayscale.prob
            ),
            'resize': transforms.Resize(
                size=(input_size, input_size)
            )
        }

        if version.parse(torch.__version__) >= version.parse('1.7.1'):
            operations['gaussian_blur'] = random_apply(
                transforms.GaussianBlur(
                    kernel_size=aug_args.gaussian_blur.kernel_size,
                ),
                p=aug_args.gaussian_blur.prob
            )

        return operations

    operations = get_operations(aug_args, cfg.network.side_input_size)
    augmentations = []
    for op in weak_data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.mean, cfg.dataset.std)
    ]
    lpm_resize = transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size))
    side_resize = transforms.Resize((cfg.network.side_input_size, cfg.network.side_input_size))

    if cfg.dataset.preload_path:
        train_preprocess = transforms.Compose([*augmentations, *normalization])
        test_preprocess = transforms.Compose([side_resize, *normalization])
    else:
        train_lpm_preprocess = transforms.Compose([lpm_resize, *normalization])
        train_side_preprocess = transforms.Compose([*augmentations, *normalization])
        test_lpm_preprocess = transforms.Compose([lpm_resize, *normalization])
        test_side_preprocess = transforms.Compose([side_resize, *normalization])

        train_preprocess = (train_lpm_preprocess, train_side_preprocess)
        test_preprocess = (test_lpm_preprocess, test_side_preprocess)

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
