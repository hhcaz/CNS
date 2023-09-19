import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import argparse
import pybullet as p
import torch
import torch.optim as optim
from datetime import datetime
from cns.utils.trainer import Trainer
from cns.models.graph_vs import GraphVS
from cns.sim.dataset import DataLoader as DataLoaderShort
from cns.sim.dataset_long import DataLoader as DataLoaderLong
from cns.train_gvs_short_seq import TrainingPipeline as TrainShort
from cns.train_gvs_long_seq import TrainingPipeline as TrainLong, scheduler_lr, log_event


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="CNS")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early-stop", type=int, default=40)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load", type=str, default="none")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def train(args):
    here = os.path.dirname(__file__)
    if args.save:
        date = datetime.now().strftime("%m_%d_%H_%M_%S_")
        save_dir = os.path.join(here, "checkpoints", date + args.name)
        log_dir = os.path.join(here, "logs", date + args.name)
        print("[INFO] Checkpoints saved to {}".format(os.path.abspath(save_dir)))
        print("[INFO] Logs saved to {}".format(os.path.abspath(log_dir)))
    else:
        save_dir = log_dir = None

    data_class = DataLoaderLong if args.long else DataLoaderShort
    env = "PointGUI" if args.gui else "Point"
    if args.gui:
        p.connect(p.GUI_SERVER)

    train_loader = data_class(None, args.batch_size, train=True, num_trajs=640, env=env)
    valid_loader = data_class(None, 64, train=False, num_trajs=64, env=env)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] Run on device: {}".format(device))
    if args.load.lower() == "none":
        model = GraphVS(
            input_dim=train_loader.num_node_features, 
            pos_dim=train_loader.num_pos_features,
            hidden_dim=args.hdim,
            regress_norm=True
        ).to(device)
    else:
        model = torch.load(args.load, map_location=device)["net"]

    pg_wi_decay, pg_wo_decay = model.get_parameter_groups()
    optimizers = {
        "0": optim.AdamW(pg_wi_decay, lr=args.init_lr, weight_decay=args.weight_decay),
        "1": optim.AdamW(pg_wo_decay, lr=args.init_lr, weight_decay=0),
    }
    trainer = Trainer(model, optimizers)
    pipeline = TrainLong() if args.long else TrainShort()

    train_options = {
        'epochs': args.epochs,
        'device': device,
        'train_loader': train_loader,
        'loss_func': pipeline.compute_train_loss,
        'valid_loader': valid_loader,
        'eval_func': pipeline.compute_eval_score,
        'early_stop_epoches': args.early_stop,
        'output_path': save_dir,
        'scheduler_func': scheduler_lr,
        'print_freq': 10 if args.long else 200,
        'log_func': log_event,
        'log_path': log_dir
    }
    trainer.start(**train_options)


if __name__ == "__main__":
    train(parse_args())
