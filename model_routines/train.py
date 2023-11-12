import os
import numpy as np
import torch 
from torch import optim
import csv

from ..utils.model_utils import load_or_initialize_training, make_dataloader, exp_decay_learning_rate, train_one_epoch
    
def train(data_dir, model, loss_functions, loss_weights, init_lr, max_epoch, training_regions='overlapping', out_dir=None, decay_rate=0.995, backup_interval=10, batch_size=1):

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    training_loss_path = os.path.join(out_dir, 'training_loss.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)
        os.system(f'chmod a+rwx {backup_ckpts_dir}')

    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Training regions: {training_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Decay rate: {decay_rate}")
    print(f"Backup interval: {backup_interval}")
    print(f"Batch size: {batch_size}")
    print("---------------------------------------------------")

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0, amsgrad=True)

    # Check if training for first time or continuing from a saved checkpoint.
    epoch_start = load_or_initialize_training(model, optimizer, latest_ckpt_path)

    train_loader = make_dataloader(data_dir, shuffle=True, mode='train', batch_size=batch_size)

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate)

        average_epoch_loss = train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions)

        # Save and report loss from the epoch.
        save_tloss_csv(training_loss_path, epoch, average_epoch_loss)
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')

        print('Saving model checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'model': model,
            'loss_functions': loss_functions,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'training_regions': training_regions,
            'decay_rate': decay_rate
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
        print('Checkpoint saved successfully.')

def save_tloss_csv(pathname, epoch, tloss):
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            writer.writerow(['Epoch', 'Training Loss'])
        writer.writerow([epoch, tloss])

if __name__ == '__main__':

    from ..models import unet3d
    import torch.nn as nn

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    model = unet3d.U_Net3d()
    loss_functions = [nn.MSELoss(), nn.CrossEntropyLoss()]
    loss_weights = [0.4, 0.7]
    lr = 6e-5
    max_epoch = 20
    out_dir = '/mmfs1/home/ehoney22/debug'

    train(data_dir, model, loss_functions, loss_weights, lr, max_epoch, out_dir=out_dir)
