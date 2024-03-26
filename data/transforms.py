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

    operations = get_operations(aug_args, cfg.data.coarse_input_size)
    augmentations = []
    for op in weak_data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ]
    fine_resize = transforms.Resize((cfg.data.fine_input_size, cfg.data.fine_input_size))
    coarse_resize = transforms.Resize((cfg.data.coarse_input_size, cfg.data.coarse_input_size))

    if cfg.base.preload:
        train_preprocess = transforms.Compose([*augmentations, *normalization])
        test_preprocess = transforms.Compose([coarse_resize, *normalization])
    else:
        train_fine_preprocess = transforms.Compose([fine_resize, *normalization])
        train_coarse_preprocess = transforms.Compose([*augmentations, *normalization])
        test_fine_preprocess = transforms.Compose([fine_resize, *normalization])
        test_coarse_preprocess = transforms.Compose([coarse_resize, *normalization])

        train_preprocess = (train_fine_preprocess, train_coarse_preprocess)
        test_preprocess = (test_fine_preprocess, test_coarse_preprocess)

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
