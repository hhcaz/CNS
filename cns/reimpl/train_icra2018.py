import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pybullet as p
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..utils.trainer import Trainer
from .dataset import DataLoader
from .icra2018 import ICRA2018


def compute_train_loss(net: ICRA2018, data, device):
    """
    Arguments:
    - net: torch.nn.Module
    - data: data from (for data in dataloader)
    - device: torch.Device
    Returns:
    - results: dict, metrics for logging
    - loss: training loss
    - do_update: bool, whether conduct gradient descent now
    """
    image_current, image_target, _, gt_poses = [d.to(device) for d in data]
    raw_pred = net({"cur_img": image_current, "tar_img": image_target})
    result = net.criterion(raw_pred, gt_poses)
    loss = result["total_loss"]

    # add prefix
    result = {("train/" + k): v.item() for k, v in result.items()}

    vel = net.postprocess(raw_pred)
    train_loader.feedback(vel * 2, norm=False)
    train_loader.resample_pose()  # comment this line to enable DAgger

    return result, loss, True


def compute_eval_score(net: ICRA2018, data, device):
    """
    Arguments:
    - net: torch.nn.Module
    - data: data from (for data in dataloader)
    - device: torch.Device
    Returns:
    - results: dict, metrics for logging
    - score: eval score
    - larger_better: bool, whether larger eval score means better model
    """
    image_current, image_target, _, gt_poses = [d.to(device) for d in data]
    raw_pred = net({"cur_img": image_current, "tar_img": image_target})
    result = net.criterion(raw_pred, gt_poses)
    loss = result["total_loss"]

    # add prefix
    result = {("valid/" + k): v.item() for k, v in result.items()}

    vel = net.postprocess(raw_pred)
    valid_loader.feedback(vel * 2, norm=False)

    return result, loss.item(), False


def scheduler_lr(optimizers, epoch, not_improve_epoches):
    if (not_improve_epoches > 0) and (not_improve_epoches % 5) == 0:
        for name, optimizer in optimizers.items():
            shrink_lr(optimizer, 0.5)
            pass


def shrink_lr(optimizer: optim.Optimizer, shrink_factor=0.5):
    before_lr = optimizer.param_groups[0]['lr']
    if before_lr * shrink_factor < 1e-5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= shrink_factor
    after_lr = optimizer.param_groups[0]['lr']
    print('[INFO] Shrink learning rate from {} to {}'.format(before_lr, after_lr))


def log_event(writer: SummaryWriter, epoch, train_result, valid_result):
    for tag, value in train_result.items():
        writer.add_scalar(tag, value, epoch)
    for tag, value in valid_result.items():
        writer.add_scalar(tag, value, epoch)


if __name__ == "__main__":
    epochs = 100
    batch_size = 16
    device = torch.device("cuda:0")

    ckpt_path = None

    here = os.path.dirname(__file__)
    date = datetime.now().strftime("%m_%d_%H_%M_%S_")
    suffix = date + "ICRA2018"
    if "posix" in os.name:
        save_path = os.path.join(here, "..", "checkpoints", suffix)
        log_path = os.path.join(here, "..", "logs", suffix)
        print("[INFO] Save path: {}".format(save_path))
    else:
        save_path = None
        log_path = None

    p.connect(p.GUI_SERVER)
    train_loader = DataLoader(None, batch_size=batch_size, train=True, num_trajs=200, env="ImageGUI")
    valid_loader = DataLoader(None, batch_size=batch_size, train=False, num_trajs=20, env="ImageGUI")

    if ckpt_path is None:
        model = ICRA2018().to(device)
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = ckpt["net"]

    optimizers = {
        "0": optim.Adam(model.trainable_parameters(), lr=5e-4)
    }

    trainer = Trainer(model, optimizers)
    train_options = {
        'epochs': epochs,
        'device': device,
        'train_loader': train_loader,
        'loss_func': compute_train_loss,
        'valid_loader': valid_loader,
        'eval_func': compute_eval_score,
        'early_stop_epoches': 100,
        'output_path': save_path,
        'scheduler_func': scheduler_lr,
        'print_freq': 200,
        'log_func': log_event,
        'log_path': log_path
    }
    trainer.start(**train_options)

