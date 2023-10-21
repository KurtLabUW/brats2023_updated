import os
import numpy as np
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
# make sure 'models' folder is in same directory for next import to run
from data import datasets, trans # in same directory, could rewrite these .py files to be cleaner too
import torch.nn as nn
# import losses as lf2 # in same directory
# import EdgeLoss3D # in same directory
from models import unet3d

LOSS_STR_TO_FUNC = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.CrossEntropyLoss(),
    # 'edge-loss': EdgeLoss3D.GMELoss3D(),
    # 'dice': lf.DiceLoss(),
    # 'focal': lf.FocalLoss()
    # 'hd'
}

MODEL_STR_TO_FUNC = {
    'unet3d': unet3d.U_Net3d()
}

def exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Exponentially decays learning rate of optimizer at given epoch."""
    lr = init_lr * (decay_rate ** (epoch-1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def seg_to_one_hot_channels(seg):
    """Converts segmentation to 3 channels, each a one-hot encoding of a tumour region label."""
    B,_,H,W,D = seg.shape
    seg3 = torch.zeros((B,3,H,W,D))
    for channel_value in [1,2,3]:
        seg3[:, channel_value-1, :, :, :] = (seg == channel_value).type(torch.float)
    return seg3

def disjoint_to_overlapping(seg):
    """Converts tensor channels from representing disjoint regions to overlapping ones."""
    mask = torch.zeros_like(seg)
    mask[:,0] = seg[:, 0] + seg[:, 1] + seg[:, 2] #WHOLE TUMOR
    mask[:,1] = seg[:, 0] + seg[:, 2] #TUMOR CORE
    mask[:,2] = seg[:, 2] #ENHANCING TUMOR
    return mask
    
def train(data_dir, model_str, loss_functs_str, loss_weights, init_lr, max_epoch, training_regions='overlapping', out_dir=None, decay_rate=0.995, save_interval=10, batch_size=1):

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    saved_ckpts_dir = os.path.join(out_dir, 'saved_ckpts')
    if not os.path.exists(saved_ckpts_dir):
        os.makedirs(saved_ckpts_dir)
        os.system(f'chmod a+rwx {saved_ckpts_dir}')

    print(f"TRAINING SUMMARY")
    print(f"Model: {model_str}")
    print(f"Loss functions: {loss_functs_str}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Training regions: {training_regions}")
    print(f"Data directory: {data_dir}")
    print(f"Out directory: {out_dir}")
    print(f"Save interval: {save_interval}")
    print("----------------------------------")

    model = MODEL_STR_TO_FUNC[model_str]
    loss_functs = [LOSS_STR_TO_FUNC[l] for l in loss_functs_str]
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0, amsgrad=True)

    # Check if training for first time or continuing from a saved checkpoint.
    if not os.path.exists(latest_ckpt_path):
        epoch_start = 1
        print('No training checkpoint found. Will train from beginning.')
    else:
        print('Training checkpoint found. Loading checkpoint...')
        checkpoint = torch.load(latest_ckpt_path)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])
        print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')

    train_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32))
                                              ])
    train_set = datasets.BratsDataset(data_dir, transforms=train_composed, mode='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

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
            loss = 0.
            for n, loss_function in enumerate(loss_functs):      
                temp = 0
                for i in range(3):
                    temp += loss_function(output[:,i:i+1].to(device='cuda:1'), seg[:,i:i+1].to(device='cuda:1'))

                loss += temp * loss_weights[n]

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
            'optim_sd': optimizer.state_dict()
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % save_interval == 0:
            torch.save(checkpoint, os.path.join(saved_ckpts_dir, f'epoch{epoch}.pth.tar'))
        print('Checkpoint saved successfully.')

if __name__ == '__main__':

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    model_str = 'unet3d'
    loss_functs_str = ['mse', 'cross-entropy']
    weights = [0.4, 0.7]
    lr = 6e-5
    max_epoch = 20
    out_dir = '/mmfs1/home/ehoney22/debug'

    train(data_dir, model_str, loss_functs_str, weights, lr, max_epoch, out_dir=out_dir)