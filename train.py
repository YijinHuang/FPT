import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.func import *
from modules.loss import *
from modules.scheduler import *


def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    device = cfg.base.device
    accelerator = cfg.accelerator
    optimizer = initialize_optimizer(cfg, model)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # start training
    model.train()
    max_indicator = 0
    for epoch in range(cfg.train.epochs):
        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if cfg.base.progress else enumerate(train_loader)
        for step, train_data in progress:
            scheduler_step = epoch + step / len(train_loader)
            lr = adjust_learning_rate(cfg, optimizer, scheduler_step)

            if cfg.base.preload:
                X_coarse, key_states, value_states, y = train_data
                key_states, value_states = key_states.to(device), value_states.to(device)
                key_states = key_states.transpose(0, 1)
                value_states = value_states.transpose(0, 1)
            else:
                X_fine, X_coarse, y = train_data
                X_fine = X_fine.to(device)
                with torch.no_grad():
                    _, key_states, value_states = frozen_encoder(X_fine, interpolate_pos_encoding=True)

            X_coarse, y = X_coarse.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)
            # forward
            y_pred = model(X_coarse, key_states, value_states)
            loss = loss_function(y_pred, y)      

            # backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if is_main(cfg):
                cfg.accelerator.gather(y_pred)
                cfg.accelerator.gather(y)

                epoch_loss += loss.item()
                avg_loss = epoch_loss / (step + 1)

                estimator.update(y_pred, y)
                message = 'epoch: [{} / {}], cls_loss: {:.6f}, lr: {:.4f}'.format(epoch + 1, cfg.train.epochs, avg_loss, lr)
                if cfg.base.progress:
                    progress.set_description(message)
            
        if is_main(cfg) and not cfg.base.progress:
            print(message)

        if is_main(cfg):
            train_scores = estimator.get_scores(4)
            scores_txt = ', '.join(['{}: {}'.format(metric, score) for metric, score in train_scores.items()])
            print('Training metrics:', scores_txt)

        if epoch % cfg.train.save_interval == 0:
            save_name = 'checkpoint_{}.pt'.format(epoch)
            save_checkpoint(cfg, model, save_name)

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            eval(cfg, frozen_encoder, model, val_loader, estimator, device)
            val_scores = estimator.get_scores(6)
            scores_txt = ['{}: {}'.format(metric, score) for metric, score in val_scores.items()]
            print_msg('Validation metrics:', scores_txt)

            # save model
            indicator = val_scores[cfg.train.indicator]
            if is_main(cfg) and indicator > max_indicator:
                save_name = 'best_validation_weights.pt'
                save_checkpoint(cfg, model, save_name)
                max_indicator = indicator

    if is_main(cfg):
        save_name = 'final_weights.pt'
        save_checkpoint(cfg, model, save_name)


def evaluate(cfg, frozen_encoder, model, test_dataset, estimator):
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory
    )

    print('Running on Test set...')
    eval(cfg, frozen_encoder, model, test_loader, estimator, cfg.base.device)

    if is_main(cfg):
        print('================Finished================')
        test_scores = estimator.get_scores(6)
        for metric, score in test_scores.items():
            print('{}: {}'.format(metric, score))
        print('Confusion Matrix:')
        print(estimator.get_conf_mat())
        print('========================================')


def eval(cfg, frozen_encoder, model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        if cfg.base.preload:
            X_coarse, key_states, value_states, y = test_data
            key_states, value_states = key_states.to(device), value_states.to(device)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)
        else:
            X_fine, X_coarse, y = test_data
            X_fine = X_fine.to(device)
            with torch.no_grad():
                _, key_states, value_states = frozen_encoder(X_fine, interpolate_pos_encoding=True)

        X_coarse, y = X_coarse.to(device), y.to(device)
        y = select_target_type(y, cfg.train.criterion)

        y_pred = model(X_coarse, key_states, value_states)
        cfg.accelerator.gather(y_pred)
        cfg.accelerator.gather(y)

        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    criterion_args = cfg.criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    nesterov = cfg.solver.nesterov
    adamw_betas = cfg.solver.adamw_betas

    parameters = model.parameters()

    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=adamw_betas,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.warmup_epochs
    learning_rate = cfg.solver.learning_rate
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler


def adjust_learning_rate(cfg, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.solver.learning_rate * epoch / cfg.train.warmup_epochs
    else:
        lr = cfg.solver.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_distill_weight(cfg, epoch):
    distill_weight = cfg.train.distill_weight * 0.5 * (1. + math.cos(math.pi * epoch / cfg.train.epochs))
    return distill_weight


def save_checkpoint(cfg, model, save_name):
    accelerator = cfg.accelerator
    accelerator.save_state(os.path.join(cfg.base.save_path, 'checkpoints'))
    state_dict = accelerator.unwrap_model(model).state_dict()
    accelerator.save(state_dict, os.path.join(cfg.base.save_path, save_name))
    print_msg('Model save at {}'.format(cfg.base.save_path))


def resume(cfg):
    accelerator = cfg.accelerator
    checkpoint_path = os.path.join(cfg.base.save_path, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        accelerator.print('Loading checkpoint {}'.format(checkpoint_path))
        accelerator.load_state(checkpoint_path)
        cfg.start_epoch = int(checkpoint_path.split('_')[1].split('.')[0])
        accelerator.print('Loaded checkpoint {} from epoch {}'.format(checkpoint_path, cfg.start_epoch))
    else:
        accelerator.print('No checkpoint found at {}'.format(checkpoint_path))
