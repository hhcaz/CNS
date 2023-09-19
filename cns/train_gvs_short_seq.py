import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pybullet as p
from datetime import datetime
import torch
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

from .utils.trainer import Trainer
from .sim.dataset import DataLoader
from .models.graph_vs import GraphVS


class TrainingPipeline(object):
    def __init__(self, teacher_epochs=0, steps_for_update=8):
        self.teacher_epochs = teacher_epochs
        self.hidden_train = None
        self.hidden_valid = None
        self.steps_train = 0
        self.steps_for_update = steps_for_update

    def compute_train_loss(self, net: GraphVS, data, device, epoch, train_loader: DataLoader):
        data: Batch = data.to(device)
        if hasattr(net, "preprocess"):
            data = net.preprocess(data)

        if getattr(data, "new_scene").any():
            self.hidden_train = None

        raw_pred = net(data, self.hidden_train)
        self.hidden_train = raw_pred[-1]

        result, loss = net.objectives(raw_pred, data)
        result = {"train/"+k: v for k, v in result.items()}  # add prefix

        if epoch < self.teacher_epochs:
            train_loader.feedback(getattr(data, "vel") * 2, norm=False)
        else:
            pred_vel = net.postprocess(raw_pred, data)
            num_gt_policy = max(1, int(pred_vel.size(0) * 0.1))
            pred_vel[:num_gt_policy] = getattr(data, "vel")[:num_gt_policy]
            train_loader.feedback(pred_vel * 2, norm=False)

        do_update = ((self.steps_train + 1) % self.steps_for_update) == 0
        if do_update:
            self.hidden_train = self.hidden_train.clone().detach()
            self.steps_train = 0
        else:
            self.steps_train += 1

        return result, loss, do_update

    def compute_eval_score(self, net: GraphVS, data, device, epoch, valid_loader):
        data: Batch = data.to(device)
        if hasattr(net, "preprocess"):
            data = net.preprocess(data)

        if getattr(data, "new_scene").any():
            self.hidden_valid = None

        raw_pred = net(data, self.hidden_valid)
        self.hidden_valid = raw_pred[-1]

        result, loss = net.objectives(raw_pred, data)
        result = {"valid/"+k: v for k, v in result.items()}

        pred_vel = net.postprocess(raw_pred, data)
        valid_loader.feedback(pred_vel * 2, norm=False)

        eval_loss = loss.item()
        if epoch < self.teacher_epochs:
            eval_loss *= 10 if eval_loss > 0 else 0.1

        return result, eval_loss, False


def scheduler_lr(optimizers, epoch, not_improve_epoches):
    if (not_improve_epoches > 0) and (not_improve_epoches % 5) == 0:
        for name, optimizer in optimizers.items():
            shrink_lr(optimizer, 0.5)
            pass


def shrink_lr(optimizer, shrink_factor=0.5):
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


def train(
    regress_norm=True,
    model_class=GraphVS, 
    data_class=DataLoader,
    epochs=160,
    device=torch.device("cuda:0"),
    batch_size=64,
    hidden_dim=128,
    ckpt_path=None,
    suffix="",
    save=False,
    gui=False,
    teacher_epochs=0,
    steps_for_update=8,
):
    here = os.path.dirname(__file__)
    date = datetime.now().strftime("%m_%d_%H_%M_%S_")
    if len(suffix): suffix = "_" + suffix
    suffix = date + "graph_{}".format(hidden_dim) + suffix

    if save:
        save_path = os.path.join(here, "checkpoints", suffix)
        log_path = os.path.join(here, "logs", suffix)
        print("[INFO] Checkpoints saved to {}".format(os.path.abspath(save_path)))
        print("[INFO] Logs saved to {}".format(os.path.abspath(log_path)))
    else:
        save_path = None
        log_path = None
    
    env = "PointGUI" if gui else "Point"
    if "gui" in env.lower():
        p.connect(p.GUI_SERVER)
    train_loader = data_class(None, batch_size, train=True, num_trajs=400, env=env)
    valid_loader = data_class(None, batch_size, train=False, num_trajs=100, env=env)

    if ckpt_path is None:
        model = model_class(
            input_dim=train_loader.num_node_features, 
            pos_dim=train_loader.num_pos_features,
            hidden_dim=hidden_dim,
            regress_norm=regress_norm
        ).to(device)

        pg_wi_decay, pg_wo_decay = model.get_parameter_groups()
        optimizers = {
            "0": optim.AdamW(pg_wi_decay, lr=5e-4, weight_decay=1e-4),
            "1": optim.AdamW(pg_wo_decay, lr=5e-4, weight_decay=0),
        }
        trainer = Trainer(model, optimizers)
    else:
        trainer = Trainer.load_from_checkpoint(ckpt_path, device)

    pipeline = TrainingPipeline(teacher_epochs, steps_for_update)
    train_options = {
        'epochs': epochs,
        'device': device,
        'train_loader': train_loader,
        'loss_func': pipeline.compute_train_loss,
        'valid_loader': valid_loader,
        'eval_func': pipeline.compute_eval_score,
        'early_stop_epoches': 40,
        'output_path': save_path,
        'scheduler_func': scheduler_lr,
        'print_freq': 200,
        'log_func': log_event,
        'log_path': log_path
    }
    trainer.start(**train_options)


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        regress_norm=True, 
        # suffix="",
        save="posix" in os.name,
        # save=True,
        gui=True,
    )
