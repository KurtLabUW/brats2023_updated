import torch
from torch import optim
import numpy as np
import os

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
# make sure 'models' folder is in same directory for next import to run
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too

from utils import *

FREEZE_STR_TO_LAYERS = {
    'encoder': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7'],
    'decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12', 'Up1', 'Up_conv1', 'Conv_1x11'],
    'middle' : ['Conv5', 'Conv6', 'Conv7', 'Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
    'none' : [],
    'deep_decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4']
}

def freeze_layers(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            print(f'Freezing parameter {name}')
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
                print(f'Warning! Param {name} should not require grad but does')
                break
            else:
                print(f'Parameter {name} is frozen')


def freeze_then_continue_training(data_dir, ckpt_A_path, max_epoch, frozen_layers, out_dir=None, save_interval=10, batch_size=1):

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    ckpt_B_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    saved_ckpts_B_dir = os.path.join(out_dir, 'saved_ckpts_B')
    if not os.path.exists(saved_ckpts_B_dir):
        os.makedirs(saved_ckpts_B_dir)
        os.system(f'chmod a+rwx {saved_ckpts_B_dir}')

    ## Previous training summary A
    ## Continue training summary B

    print(f"Loading model from {ckpt_A_path}...")
    checkpoint = torch.load(ckpt_A_path)

    # epoch = checkpoint['epoch']
    training_regions = checkpoint['training_regions']
    model_str = checkpoint['model_str']
    model = MODEL_STR_TO_FUNC[model_str]
    model.load_state_dict(checkpoint['model_sd'])
    loss_functs_str = checkpoint['loss_functs_str']
    loss_functs = [LOSS_STR_TO_FUNC[l] for l in loss_functs_str]
    loss_weights = checkpoint['loss_weights']
    
    init_lr = checkpoint['init_lr']
    decay_rate = checkpoint['decay_rate']

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0, amsgrad=True)

    # Check if continuing training for first time or continuing from a saved checkpoint.
    if not os.path.exists(ckpt_B_path):
        epoch_start = 1
        print('No continuation of training checkpoint found. Will start continuation of train from beginning.')

        freeze_layers(model, frozen_layers) 

    else:
        print("Continuation of training checkpoint found. Loading checkpoint...")
        checkpoint = torch.load(ckpt_B_path)

        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])

        freeze_layers(model, frozen_layers) 

    train_loader = make_dataloader(data_dir, shuffle=True, mode='train')

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        if epoch < 5:
            check_frozen(model, frozen_layers)

        exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate)

        losses_over_epoch = []
        for _, imgs, seg in train_loader:

            model.train()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs]
            seg = seg.cuda()

            # Split segmentation into 3 channels.
            seg = seg_to_one_hot_channels(seg)

            if training_regions == 'overlapping':
                seg = disjoint_to_overlapping(seg)

            x_in = torch.cat(imgs, dim=1)
            output = model(x_in)
            output = output.float()

            # Compute weighted loss, summed across each region.

            loss = compute_loss(output, seg, loss_functs, loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_over_epoch.append(loss.detach().cpu())

        # Compute, save and report loss from the epoch.
        average_epoch_loss = np.mean(losses_over_epoch)
        with open(os.path.join(out_dir, 'loss_values.dat'), 'a') as f:
            f.write(f'{epoch}, {average_epoch_loss}\n')
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')

        print('Saving model checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'model_str': model_str,
            'training_regions': training_regions,
            'loss_functs_str': loss_functs_str,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'decay_rate': decay_rate
        }
        torch.save(checkpoint, ckpt_B_path)
        if epoch % save_interval == 0:
            torch.save(checkpoint, os.path.join(saved_ckpts_B_dir, f'epoch{epoch}.pth.tar'))
        print('Checkpoint saved successfully.')


if __name__ == '__main__':

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    ckpt_A_path = '/mmfs1/home/ehoney22/debug/saved_ckpts/epoch20.pth.tar'
    max_epoch = 20
    frozen_layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7']
    out_dir = '/mmfs1/home/ehoney22/debug/freeze_then_continue_training'

    freeze_then_continue_training(data_dir, ckpt_A_path, max_epoch, frozen_layers, out_dir)