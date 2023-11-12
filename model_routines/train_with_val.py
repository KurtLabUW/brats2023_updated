import os
import numpy as np
import torch 
from torch import optim
import csv
from monai.metrics import DiceMetric

from ..utils.model_utils import load_or_initialize_training, make_dataloader, exp_decay_learning_rate, compute_loss
from ..utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping, probs_to_preds

def train_with_val(train_data_dir, val_data_dir, model, loss_functions, loss_weights, init_lr, max_epoch, training_regions='overlapping', eval_regions='overlapping', out_dir=None, decay_rate=0.995, backup_interval=10, batch_size=1):

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    best_vloss_ckpt_path = os.path.join(out_dir, 'best_vloss_ckpt.pth.tar')
    best_dice_ckpt_path = os.path.join(out_dir, 'best_dice_ckpt.pth.tar')
    loss_and_metrics_path = os.path.join(out_dir, 'loss_and_metrics.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)
        os.system(f'chmod a+rwx {backup_ckpts_dir}')

    print("---------------------------------------------------")
    print(f"TRAINING WITH VALIDATION SUMMARY")
    print(f"Training data directory: {train_data_dir}")
    print(f"Validation data directory: {val_data_dir}")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Training regions: {training_regions}")
    print(f"Evaluation regions: {eval_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Decay rate: {decay_rate}")
    print(f"Backup interval: {backup_interval}")
    print(f"Batch size: {batch_size}")
    print("---------------------------------------------------")

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0, amsgrad=True)

    # Check if training for first time or continuing from a saved checkpoint.
    epoch_start, best_vloss, best_dice = load_or_initialize_training(model, optimizer, latest_ckpt_path, train_with_val=True)

    train_loader = make_dataloader(train_data_dir, shuffle=True, mode='train', batch_size=batch_size)
    val_loader = make_dataloader(val_data_dir, shuffle=False, mode='train', batch_size=batch_size)

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate)

        losses_over_epoch = []
        for _, imgs, seg in train_loader:

            model.train()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs] # img is B1HWD
            seg = seg.cuda()

            # Split segmentation into 3 channels.
            seg = seg_to_one_hot_channels(seg)
            # seg is B3HWD - each channel is one-hot encoding of a disjoint region

            if training_regions == 'overlapping':
                seg = disjoint_to_overlapping(seg)
                # seg is B3HWD - each channel is one-hot encoding of an overlapping region

            x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
            output = model(x_in)
            output = output.float()

            # Compute weighted loss, summed across each training region.
            loss = compute_loss(output, seg, loss_functions, loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_over_epoch.append(loss.detach().cpu())

        # Compute, save and report loss from the epoch.
        average_epoch_loss = np.mean(losses_over_epoch)
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')

        val_loss_vals = []

        # Recommend use MONAI metrics set-up for different metrics (Cumulative Iterative)
        dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

        with torch.no_grad():
            for _, imgs, seg in val_loader:

                model.eval()

                # Move data to GPU.
                imgs = [img.cuda() for img in imgs] # img is B1HWD
                seg = seg.cuda()

                # Split segmentation into 3 channels.
                seg = seg_to_one_hot_channels(seg) # seg is B3HWD

                if training_regions == 'overlapping':
                    seg_train = disjoint_to_overlapping(seg)
                    # seg_train is B3HWD - each channel is one-hot encoding of an overlapping region
                elif training_regions == 'disjoint':
                    seg_train = seg
                    # seg_train is B3HWD - each channel is one-hot encoding of a disjoint region

                x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
                output = model(x_in)
                output = output.float()

                # Compute weighted loss, summed across each training region.
                val_loss = compute_loss(output, seg_train, loss_functions, loss_weights)
                val_loss_vals.append(val_loss.detach().cpu())

                preds = probs_to_preds(output, training_regions)

                eval_region_names = []
                if eval_regions == 'overlapping':
                    # eval_region_names = ['WT', 'TC', 'ET']
                    # Convert seg and pred to 3 channels corresponding to overlapping regions
                    seg_eval = disjoint_to_overlapping(seg)
                    preds_eval = disjoint_to_overlapping(preds)
                    
                elif eval_regions == 'disjoint':
                    # eval_region_names = ['NCR', 'ED', 'ET']
                    # Convert seg and pred to 3 channels corresponding to disjoint regions
                    seg_eval = seg
                    preds_eval = preds

                # seg_eval is B3HWD
                # preds_eval is B3HWD

                # Compute metrics between seg_eval and preds_eval.
                dice_metric(y_pred = preds_eval, y=seg_eval)

        # Compute and report validation loss.
        average_val_loss = np.mean(val_loss_vals)
        print(f'Validation completed. Average validation loss = {average_val_loss}')

        # Aggregate and report the Dice scores.
        dice_metric_batch = dice_metric.aggregate()
        eval_region_dice_scores = []
        for i in range(3):
            eval_region_dice_scores.append(dice_metric_batch[i].item())
        mean_dice = np.mean(eval_region_dice_scores)

        update_vloss = False
        if average_val_loss < best_vloss:
            best_vloss = average_val_loss
            update_vloss = True

        update_dice = False
        if mean_dice > best_dice:
            best_dice = mean_dice
            update_dice = True

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
            'decay_rate': decay_rate,
            'vloss': best_vloss,
            'dice': best_dice
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
        if update_vloss:
            print('New best validation loss!')
            torch.save(checkpoint, best_vloss_ckpt_path)
        if update_dice:
            print('New best dice score!')
            torch.save(checkpoint, best_dice_ckpt_path)

        save_loss_and_metrics_csv(loss_and_metrics_path, epoch, average_epoch_loss, average_val_loss, mean_dice, eval_region_dice_scores, eval_region_names)

        print('Checkpoint saved successfully.')

def save_loss_and_metrics_csv(pathname, epoch, tloss, vloss, mean_dice, eval_region_scores):
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Mean Dice', 'Dice 1', 'Dice 2', 'Dice 3'])
        writer.writerow([epoch, tloss, vloss, mean_dice] + eval_region_scores)

if __name__ == '__main__':

    from ..models import unet3d
    import torch.nn as nn

    train_dir = '/mmfs1/home/ehoney22/debug_data/train'
    val_dir = '/mmfs1/home/ehoney22/debug_data/train'
    model = unet3d.U_Net3d()
    loss_functions = [nn.MSELoss(), nn.CrossEntropyLoss()]
    loss_weights = [0.4, 0.7]
    lr = 6e-5
    max_epoch = 20
    out_dir = '/mmfs1/home/ehoney22/debug/train_with_val'

    train_with_val(train_dir, val_dir, model, loss_functions, loss_weights, lr, max_epoch, out_dir=out_dir)