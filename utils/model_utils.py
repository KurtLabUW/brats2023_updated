import torch
from torch.utils.data import DataLoader
from datasets import brats_dataset
import os

def load_or_initialize_training(model, optimizer, latest_ckpt_path):
    """
    Load the training checkpoint if it exists, or initialize training from the beginning.

    Parameters:
    - model: The PyTorch model to be trained.
    - optimizer: The optimizer used for training.
    - latest_ckpt_path: The path to the latest checkpoint file.

    Returns:
    - epoch_start: The epoch to start training from.
    """

    if not os.path.exists(latest_ckpt_path):
        epoch_start = 1
        print('No training checkpoint found. Will start training from scratch.')
    else:
        print('Training checkpoint found. Loading checkpoint...')
        checkpoint = torch.load(latest_ckpt_path)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])
        print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')

    return epoch_start

def make_dataloader(data_dir, shuffle, mode, batch_size=1):
    dataset = brats_dataset.BratsDataset(data_dir, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)
    return dataloader

def exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Exponentially decays learning rate of optimizer at given epoch."""
    lr = init_lr * (decay_rate ** (epoch-1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_loss(output, seg, loss_functs, loss_weights):
    # Compute weighted loss, summed across each region.
    loss = 0.
    for n, loss_function in enumerate(loss_functs):      
        temp = 0
        for i in range(3):
            temp += loss_function(output[:,i:i+1].cuda(), seg[:,i:i+1].cuda())

        loss += temp * loss_weights[n]
    return loss

def freeze_layers(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            print(f'Freezing parameter {name}.')
            param.requires_grad = False

def check_frozen(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            if param.requires_grad:
                print(f'Warning! Param {name} should not require grad but does.')
                break
            else:
                print(f'Parameter {name} is frozen.')