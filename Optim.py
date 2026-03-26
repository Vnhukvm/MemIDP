"""
Learning rate scheduler wrapper with warm-up scheduling.
Implements the warm-up and decay schedule from "Attention Is All You Need".
"""
import numpy as np

class ScheduledOptim(object):
    """
    A wrapper class that combines an optimizer with warm-up learning rate scheduling.
    Implements the inverse square root decay schedule with warm-up steps.
    """
    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        Initialize the scheduled optimizer.
        
        Args:
            optimizer: The inner optimizer to wrap (typically AdamW).
            d_model: Model dimension used for scaling the learning rate.
            n_warmup_steps: Number of warm-up steps before decay.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        """Take a step with the inner optimizer."""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer."""
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        """
        Update learning rate based on current step.
        Follows the formula from the Transformer paper:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        This implements:
        1. Linear warm-up phase during warmup steps
        2. Inverse square root decay after warm-up
        """
        self.n_current_steps += 1

        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
