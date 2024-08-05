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
    optimizer = initialize_optimizer(cfg, model)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    # check resume
    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer)

    # start training
    model.train()
    max_indicator = 0
    for epoch in range(start_epoch, cfg.train.epochs):
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

            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = train_data
                key_states, value_states = key_states.to(device), value_states.to(device)
                key_states = key_states.transpose(0, 1)
                value_states = value_states.transpose(0, 1)
            else:
                X_lpm, X_side, y = train_data
                X_lpm = X_lpm.to(device)
                with torch.no_grad():
                    _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

            X_side, y = X_side.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            # forward
            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y)      

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)

            estimator.update(y_pred, y)
            message = 'epoch: [{} / {}], cls_loss: {:.6f}, lr: {:.4f}'.format(epoch + 1, cfg.train.epochs, avg_loss, lr)
            if cfg.base.progress:
                progress.set_description(message)
            
        if not cfg.base.progress:
            print(message)

        train_scores = estimator.get_scores(4)
        scores_txt = ', '.join(['{}: {}'.format(metric, score) for metric, score in train_scores.items()])
        print('Training metrics:', scores_txt)

        if epoch % cfg.train.save_interval == 0:
            save_name = 'checkpoint.pt'
            save_checkpoint(cfg, model, epoch, optimizer, save_name)

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            eval(cfg, frozen_encoder, model, val_loader, estimator, device)
            val_scores = estimator.get_scores(6)
            scores_txt = ['{}: {}'.format(metric, score) for metric, score in val_scores.items()]
            print_msg('Validation metrics:', scores_txt)

            # save model
            indicator = val_scores[cfg.train.indicator]
            if indicator > max_indicator:
                save_name = 'best_validation_weights.pt'
                save_weights(cfg, model, save_name)
                max_indicator = indicator

    save_name = 'final_weights.pt'
    save_weights(cfg, model, save_name)


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
        if cfg.dataset.preload_path:
            X_side, key_states, value_states, y = test_data
            key_states, value_states = key_states.to(device), value_states.to(device)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)
        else:
            X_lpm, X_side, y = test_data
            X_lpm = X_lpm.to(device)
            with torch.no_grad():
                _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

        X_side, y = X_side.to(device), y.to(device)
        y = select_target_type(y, cfg.train.criterion)

        y_pred = model(X_side, key_states, value_states)
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
        loss = nn.CrossEntropyLoss(weight=weight)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss()
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss()
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss()
    elif criterion == 'kappa_loss':
        loss = KappaLoss()
    elif criterion == 'focal_loss':
        loss = FocalLoss()
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    parameters = model.parameters()
    solver = cfg.solver.optimizer
    if solver == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.dataset.learning_rate,
            momentum=cfg.solver.momentum,
            nesterov=cfg.solver.momentum,
            weight_decay=cfg.solver.weight_decay
        )
    elif solver == 'ADAM':
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.dataset.learning_rate,
            betas=cfg.solver.betas,
            weight_decay=cfg.solver.weight_decay
        )
    elif solver == 'ADAMW':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg.dataset.learning_rate,
            betas=cfg.solver.betas,
            weight_decay=cfg.solver.weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


def adjust_learning_rate(cfg, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.dataset.learning_rate * epoch / cfg.train.warmup_epochs
    else:
        lr = cfg.dataset.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(cfg, model, epoch, optimizer, save_name):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    checkpoint_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(checkpoint, checkpoint_path)


def save_weights(cfg, model, save_name):
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print_msg('Model saved at {}'.format(save_path))


def resume(cfg, model, optimizer):
    checkpoint = cfg.base.checkpoint
    if os.path.exists(checkpoint):
        print_msg('Loading checkpoint {}'.format(checkpoint))

        checkpoint = torch.load(checkpoint, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint, checkpoint['epoch']))
        return start_epoch
    else:
        print_msg('No checkpoint found at {}'.format(checkpoint))
        raise FileNotFoundError('No checkpoint found at {}'.format(checkpoint))
