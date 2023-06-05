import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.criterion_best = 0
        self.delta = delta
        self.checkpoint_path = ''

    def __call__(self, criterion, model, epoch, number, probas=None, local_rank=None):
        score = criterion
        if self.best_score is None:
            self.best_score = score
            self.checkpoint_path = self.save_checkpoint(criterion, model, epoch, number, probas)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if (local_rank is None) or (local_rank == 0):
                self.checkpoint_path = self.save_checkpoint(criterion, model, epoch, number, probas)
            self.counter = 0
        return self.checkpoint_path

    def save_checkpoint(self, criterion, model, epoch, number, probas=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation f1 augment ({self.criterion_best:.6f} --> {criterion:.6f}).  Saving model ...')
        state = {'model': model.state_dict(), 'epoch': epoch, 'criterion': criterion, 'probas': probas}
        self.checkpoint_path = f'../results/check_{number}/checkpoint_epoch_{epoch}.pt'
        torch.save(state, self.checkpoint_path)
        self.criterion_best = criterion
        return self.checkpoint_path
