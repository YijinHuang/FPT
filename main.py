import os
import random

import hydra
import numpy as np
from omegaconf import OmegaConf

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model, load_weights


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # print configuration
    print_msg('LOADING CONFIG FILE')
    print(OmegaConf.to_yaml(cfg))

    # create folder
    save_path = cfg.dataset.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg('Save path {} exists and will be overwrited.'.format(save_path), warning=True)
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.dataset.save_path = new_save_path
            warning = 'Save path {} exists. New save path is set to be {}.'.format(save_path, new_save_path)
            print_msg(warning, warning=True)

    os.makedirs(cfg.dataset.save_path, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.dataset.save_path, 'cfg.yaml'))

    # check preloading
    if cfg.dataset.preload_path:
        assert os.path.exists(cfg.dataset.preload_path), 'Preload path does not exist.'
        print_msg('Preloading is enabled using {}'.format(cfg.dataset.preload_path))

    if cfg.base.random_seed >= 0:
        set_seed(cfg.base.random_seed, cfg.base.cudnn_deterministic)

    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    frozen_encoder, model = generate_model(cfg)
    estimator = Estimator(cfg.train.metrics, cfg.dataset.num_classes, cfg.train.criterion)
    train(
        cfg=cfg,
        frozen_encoder=frozen_encoder,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator
    )

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.dataset.save_path, 'final_weights.pt')
    load_weights(model, checkpoint)
    evaluate(cfg, frozen_encoder, model, test_dataset, estimator)

    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.dataset.save_path, 'best_validation_weights.pt')
    load_weights(model, checkpoint)
    evaluate(cfg, frozen_encoder, model, test_dataset, estimator)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == '__main__':
    main()
