"""
Early stopping implementation for model regularization.
Stops training when validation loss doesn't improve for a given number of epochs.
Saves the best model checkpoint when validation loss improves.
"""
import numpy as np
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Implements checkpoint saving when best performance is achieved.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Initialize EarlyStopping.
        
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
                Default: 7
            verbose (bool): If True, prints progress messages. Default: False
            delta (float): Minimum improvement required to qualify as an improvement.
                Default: 0
            path (str): File path to save best model checkpoint. Default: 'checkpoint.pt'
            trace_func (function): Function to print trace messages. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Check if validation loss has improved and update early stopping state.
        
        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): PyTorch model to save if improvement occurs.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint when validation loss decreases.
        
        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): PyTorch model to save.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

