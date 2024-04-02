import os

from munch import munchify
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model, load_weights


def main():
    args = parse_config()
    cfg = load_config(args.config)
    cfg = munchify(cfg)

    # print configuration
    print_msg('LOADING CONFIG FILE: {}'.format(args.config))
    print_config({
        'BASE CONFIG': cfg.base,
        'DATA CONFIG': cfg.data,
        'TRAIN CONFIG': cfg.train
    })

    # create folder
    save_path = cfg.base.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg('Save path {} exists and will be overwrited.'.format(save_path), warning=True)
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.base.save_path = new_save_path
            warning = 'Save path {} exists. New save path is set to be {}.'.format(save_path, new_save_path)
            print_msg(warning, warning=True)

    # check preloading
    cfg.base.preload = args.preload
    cfg.base.preload_path = args.preload_path
    if cfg.base.preload and not cfg.base.preload_path:
        error_message = 'Preloading is enabled but preload_path is not set. Please use --preload-path to specify the path or disable preloading.'
        raise ValueError(error_message)
    if cfg.base.preload_path and not cfg.base.preload:
        error_message = 'Preloading path is set but preloading is disabled. Please use --preload to enable preloading.'
        raise ValueError(error_message)

    os.makedirs(cfg.base.save_path, exist_ok=True)
    copy_config(args.config, cfg.base.save_path)

    if cfg.base.random_seed >= 0:
        set_seed(cfg.base.random_seed)

    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16'
    )
    cfg.base.device = accelerator.device
    cfg.accelerator = accelerator

    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    frozen_encoder, model = generate_model(cfg)
    estimator = Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)
    train(
        cfg=cfg,
        frozen_encoder=frozen_encoder,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator
    )

    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
    load_weights(model, checkpoint)
    evaluate(cfg, frozen_encoder, model, test_dataset, estimator)

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.base.save_path, 'final_weights.pt')
    load_weights(model, checkpoint)
    evaluate(cfg, frozen_encoder, model, test_dataset, estimator)


if __name__ == '__main__':
    main()
