import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pybullet as p
from datetime import datetime
import torch
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

from .utils.trainer import Trainer
from .sim.dataset_long import DataLoader
from .models.graph_vs import GraphVS


class TrainingPipeline(object):
    def __init__(self, teacher_epochs=0):
        self.teacher_epochs = teacher_epochs

    def compute_train_loss(self, net: GraphVS, loader: DataLoader, device, epoch):
        loss_seq = []
        result_seq = []

        hidden = None
        for _ in range(200):
            data: Batch = loader.get().to(device)
            if hasattr(net, "preprocess"):
                data = net.preprocess(data)
            
            raw_pred = net(data, hidden)
            hidden = raw_pred[-1]

            result, loss = net.objectives(raw_pred, data)
            result_seq.append(result)
            loss_seq.append(loss)

            if epoch < self.teacher_epochs:
                loader.feedback(getattr(data, "vel") * 2, norm=False)
            else:
                pred_vel = net.postprocess(raw_pred, data)
                num_gt_policy = max(1, int(pred_vel.size(0) * 0.1))
                pred_vel[:num_gt_policy] = getattr(data, "vel")[:num_gt_policy]
                loader.feedback(pred_vel * 2, norm=False)
            
            if loader.need_reinit_all():
                loader.reinit_all()
                hidden = None

        loss = sum(loss_seq) / len(loss_seq)
        result = {"train/"+k: sum(r[k] for r in result_seq)/len(result_seq) for k in result}
        print("[INFO]", result)

        return result, loss, True


    def compute_eval_score(self, net: GraphVS, loader: DataLoader, device, epoch):
        loss_seq = []
        result_seq = []

        hidden = None
        for _ in range(200):
            data: Batch = loader.get().to(device)
            if hasattr(net, "preprocess"):
                data = net.preprocess(data)
            
            raw_pred = net(data, hidden)
            hidden = raw_pred[-1]

            result, loss = net.objectives(raw_pred, data)
            result_seq.append(result)
            loss_seq.append(loss)

            pred_vel = net.postprocess(raw_pred, data)
            loader.feedback(pred_vel * 2, norm=False)
            
            if loader.need_reinit_all():
                loader.reinit_all()
                hidden = None
        
        loss = sum(loss_seq) / len(loss_seq)
        result = {"valid/"+k: sum(r[k] for r in result_seq)/len(result_seq) for k in result}

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
    train_loader = data_class(None, batch_size, train=True, num_trajs=640, env=env)
    valid_loader = data_class(None, 64, train=False, num_trajs=64, env=env)

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
        trainer = Trainer.load_from_checkpoint(ckpt_path, device, ignore_score=True)
        # override saved optimizer
        model = trainer.net
        pg_wi_decay, pg_wo_decay = model.get_parameter_groups()
        trainer.optimizers = {
            "0": optim.AdamW(pg_wi_decay, lr=1e-4, weight_decay=1e-4),
            "1": optim.AdamW(pg_wo_decay, lr=1e-4, weight_decay=0),
        }


    pipeline = TrainingPipeline(teacher_epochs)
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
        'print_freq': 10,
        'log_func': log_event,
        'log_path': log_path
    }
    trainer.start(**train_options)


if __name__ == "__main__":
    train(
        # ckpt_path="checkpoints/05_16_04_41_58_graph_128_wo_gn/checkpoint_last.pth",
        ckpt_path="checkpoints/08_04_18_11_27_graph4_128_be/checkpoint_best.pth",
        device=torch.device("cuda:0"),
        regress_norm=True, 
        # save="posix" in os.name,
        save=True,
        gui=True,

        batch_size=8,
        teacher_epochs=1
    )
