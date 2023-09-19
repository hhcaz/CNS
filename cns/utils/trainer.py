import os
import time
import tqdm
import torch
import inspect
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint


class AverageMeter(object):
    def __init__(self, beta=0.9):
        self._beta = beta
        self._history = []
        self._exp_avg = 0
        self._count = 0
    
    def reset(self):
        self._history.clear()
        self._exp_avg = 0
        self._count = 0
    
    def update(self, value):
        self._history.append(value)
        self._count += 1
        self._exp_avg = self._beta * self._exp_avg + (1 - self._beta) * value
    
    def __len__(self):
        return len(self._history)
    
    @property
    def avg(self):
        if len(self._history) == 0:
            return 0.0
        return np.mean(self._history)
    
    @property
    def exp_avg(self):
        return self._exp_avg
    
    @property
    def exp_avg_biased(self):
        return self._exp_avg / (1 - self._beta**self._count)


def train(net, optimizers:dict, dataloader, device, loss_func, parent_scope: dict, print_freq=10):
    """
    Arguments:
    - net: nn.Module
    - optimizers: dict of torch.optim.Optimizer
    - dataloader: Dataloader of train dataset
    - device: torch.device
    - loss_func: callable object to compute loss
        loss_func recevies 3 argumnets (the network, data from dataloader, and device),
        and returns the loss. See Trainer.start() for more info.
    - print_freq: int, print the training information after several batches
    """
    total_batches = len(dataloader)

    train_result     = {}
    loss_recorder    = AverageMeter()
    data_load_time   = AverageMeter()
    batch_total_time = AverageMeter()

    extra_kwds = {}
    required_args = inspect.getfullargspec(loss_func).args
    for key in parent_scope:
        if key in required_args:
            extra_kwds[key] = parent_scope[key]

    net.train()
    accum_loss = 0
    start_time_stamp = time.time()
    for i, data in enumerate(dataloader):
        data_load_time.update(time.time() - start_time_stamp)
        
        result, loss, do_update = loss_func(net, data, device, **extra_kwds)
        if torch.isnan(loss).any():
            print("[INFO] Find nan in loss, skip")
            continue

        loss_recorder.update(loss.item())
        accum_loss = accum_loss + loss

        for key, value in result.items():
            if key not in train_result:
                train_result[key] = AverageMeter()
            train_result[key].update(value)

        if do_update:
            # print("accum loss = {:.3f}".format(accum_loss.item()))
            for name, optimizer in optimizers.items():
                optimizer.zero_grad()
            
            accum_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            for name, optimizer in optimizers.items():
                optimizer.step()
            accum_loss = 0
        
        batch_total_time.update(time.time() - start_time_stamp)
        start_time_stamp = time.time()

        if (print_freq > 0) and ((i+1) % print_freq == 0):
            print('[INFO] Batch {}/{} | Avg data load time: {:.4f} s | Avg batch time: {:.4f} s | Avg loss: {:.5f}'
                .format(i+1, total_batches, data_load_time.avg, batch_total_time.avg, loss_recorder.avg))
    
    train_result = {k:v.avg for k, v in train_result.items()}
    return train_result, loss_recorder.avg


def valid(net, dataloader, device, eval_func, parent_scope):
    valid_result  = {}
    score_recoder = AverageMeter()

    extra_kwds = {}
    required_args = inspect.getfullargspec(eval_func).args
    for key in parent_scope:
        if key in required_args:
            extra_kwds[key] = parent_scope[key]

    net.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            result, score, larger_better = eval_func(net, data, device, **extra_kwds)
            score_recoder.update(score)

            for key, value in result.items():
                if key not in valid_result:
                    valid_result[key] = AverageMeter()
                valid_result[key].update(value)
    
    valid_result = {k:v.avg for k, v in valid_result.items()}
    return valid_result, score_recoder.avg, larger_better


def save_checkpoint(epoch, net:nn.Module, optimizers:dict, score, is_best, not_improve_epoches, save_folder):
    save_state_dict = False
    for m in net.modules():
        if isinstance(m, torch.jit.ScriptModule):
            save_state_dict = True
            break

    checkpoint = {
        'epoch': epoch,
        'net': net if not save_state_dict else net.state_dict(),
        'optimizers': optimizers,
        'score': score,
        'is_best': is_best,
        'not_improve_epochs': not_improve_epoches
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_path = os.path.join(save_folder, 'checkpoint_last.pth')
    torch.save(checkpoint, file_path)

    if is_best:
        file_path = os.path.join(save_folder, 'checkpoint_best.pth')
        torch.save(checkpoint, file_path)


class Trainer(object):
    def __init__(self, net, optimizers:dict):
        self.net = net
        self.optimizers = optimizers

        self.start_epoch = 0
        self.current_epoch = 0
        self.not_improve_epochs = 0

        self.current_score = None
        self.best_score = None
        self.is_best = False
    
    def start(self,
        epochs=100,
        device=None,
        train_loader=None,
        loss_func=None,
        valid_loader=None,
        eval_func=None,
        early_stop_epoches=30,
        output_path=None,
        scheduler_func=None,
        print_freq=10,
        log_func=None,
        log_path=None
    ):
        """
        Arguments:
        - epochs: int, epochs to train the network
        - device: torch.device, cpu or cuda, set None to use gpu if gpu is available else cpu
        - train_loader: dataloader of train dataset
        - loss_func: callable object to compute loss
            loss_func receives 3 arguments (the network, data from dataloader, and device),
            and returns the train result and loss
            Example:
                def loss_func(net:nn.Module, data:tuple, device:torch.device):
                    image, truth = data
                    image = image.to(device)
                    truth = truth.to(device)
                    output = net(image)
                    loss = compute_mse_loss(output, truth)
                    do_update = True
                    train_result = {
                        "loss": float(loss)
                    }
                    return train_result, loss, do_update
        
        - valid_loader: dataloader of valid dataset
        - eval_func: callable object to compute eval score
            eval_func receives 3 arguments (the network, data from dataloader, and device),
            and returns valid result, eval score and a bool variable indicating whether the larger score is better
            Example:
                def eval_func(net:nn.Module, data:tuple, device:torch.device):
                    image, truth = data
                    image = image.to(device)
                    truth = truth.to(device)
                    output = net(image)
                    loss = compute_loss(output, truth)
                    score = compute_F1_score(output, truth)
                    accuracy = compute_accuracy(output, truth)
                    valid_result = {
                        "loss": float(loss),
                        "score": score,
                        "accuracy": accuracy
                    }
                    return valid_result, score, True
        
        - early_stop_epoches: int, stop training if the performance doesn't improve for certain epochs
        - output_path: str, folder path to save checkpoints, set None to disable saving
        - scheduler_func: callable object to scheduler learning rate, set None to ignore
            scheduler_func receives 3 arguments (the optimizers, current epoch, not improve epochs)
            Example:
                def scheduler_func(optimizers:dict, epoch:int, not_improve_epochs:int):
                    if not_improve_epochs > 0 and not_improve_epochs % 8 == 0:
                        for name, optimizer in optimizers.items():
                            shrink_learning_rate(optimizer)
        - print_freq: int, print the training information after several batches, set negtive or zero to disable
        - log_func: callable object to handle log event
            log_func receives 5 arguments (the writer, current epoch, train result, valid result, and network)
        - log_path: tensorboard output path, set None to disable saving
        """

        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        assert train_loader is not None, 'Train dataloader should be provided'

        if (log_func is not None) and (log_path is not None):
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_path, flush_secs=60)
        else:
            writer = None

        for i in range(self.start_epoch, self.start_epoch + epochs):
            local_scope = {
                "epoch": i,
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "not_improve_epochs": self.not_improve_epochs,
                "writer": writer
            }

            # train
            print('[INFO] Epoch: {}/{} | Start training...'.format(i+1, self.start_epoch + epochs))
            train_result, avg_loss = train(
                self.net, self.optimizers, train_loader, device, loss_func, local_scope, print_freq)

            # valid
            if valid_loader is None or eval_func is None:
                print('[INFO] valid_loader or eval_func not provided, use train loss to evaluate.')
                valid_result, score, larger_better = {}, avg_loss, False
            else:
                print('[INFO] Start validating...')
                valid_result, score, larger_better = valid(
                    self.net, valid_loader, device, eval_func, local_scope)
            
            self.current_epoch = i + 1
            self.current_score = score

            if self.best_score is None:
                self.best_score = score
                self.is_best = True
                self.not_improve_epochs = 0
            else:
                if larger_better:
                    if self.best_score >= score:
                        self.not_improve_epochs += 1
                        self.is_best = False
                    else:
                        self.not_improve_epochs = 0
                        self.is_best = True
                        self.best_score = score
                else:
                    if self.best_score <= score:
                        self.not_improve_epochs += 1
                        self.is_best = False
                    else:
                        self.not_improve_epochs = 0
                        self.is_best = True
                        self.best_score = score
            
            if output_path is not None:
                self.save_to_folder(output_path)
            print('[INFO] Epoch: {}/{} | Eval score: {:.5f} | Best eval score: {:.5f}'
                .format(i+1, self.start_epoch + epochs, score, self.best_score))
            
            if self.not_improve_epochs > 0:
                print('[INFO] epochs since last improvement: {}'.format(self.not_improve_epochs))

            if self.not_improve_epochs >= early_stop_epoches:
                break
            if scheduler_func is not None:
                scheduler_func(self.optimizers, i, self.not_improve_epochs)

            if (log_func is not None) and (log_path is not None):
                log_func(writer, i, train_result, valid_result)
            
            print("[INFO] Epoch: {}/{}, other train-eval metrics:".format(i+1, self.start_epoch + epochs))
            pprint(train_result)
            pprint(valid_result)
            print('------------------------------------------------------------------')
    
    def save_to_folder(self, folder):
        save_checkpoint(self.current_epoch, self.net, self.optimizers,
                        self.current_score, self.is_best, self.not_improve_epochs, folder)
    
    @classmethod
    def load_from_checkpoint(cls, file_path, map_location=None, ignore_score=False):
        checkpoint = torch.load(file_path, map_location)
        net = checkpoint['net']
        optimizers = checkpoint['optimizers']

        trainer = cls(net, optimizers)
        trainer.start_epoch        = checkpoint['epoch']
        trainer.not_improve_epochs = checkpoint['not_improve_epochs']

        if not ignore_score:
            trainer.current_score = checkpoint['score']
            trainer.best_score    = checkpoint['score']
            trainer.is_best       = checkpoint['is_best']
        
        return trainer
