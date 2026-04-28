import numpy as np

class ConstantLRScheduler:
    def __init__(self, lr):
        self.lr = lr
    
    def get_lr(self):
        return self.lr
    
    def step(self):
        pass

class CosineAnnealingLRScheduler:
    def __init__(self, eta_max, eta_min, T_max, T_mult=1):
        """
        Cosine Annealing Learning Rate Scheduler.
        
        Args:
            eta_max (float): Maximum learning rate.
            eta_min (float): Minimum learning rate.
            T_max (int): Number of steps per half-cycle (cosine decay period).
            T_mult (float): Multiplier for T_max after each restart (default: 1, no increase).
        """
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = T_max
        self.T_mult = T_mult
        self.current_step = 0
        self.current_T_max = T_max
        self.cycle = 0

    def get_lr(self):
        """
        Compute the learning rate for the current step.
        
        Returns:
            float: Current learning rate.
        """
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (self.current_step % self.current_T_max) / self.current_T_max))
        lr = self.eta_min + (self.eta_max - self.eta_min) * cosine_decay
        return lr

    def step(self):
        """
        Increment the step and update the scheduler.
        Handles warm restarts if T_mult > 1.
        """
        self.current_step += 1
        if self.current_step % self.current_T_max == 0:
            self.cycle += 1
            self.current_T_max = self.T_max * (self.T_mult ** self.cycle)
            self.current_step = 0

class StepLRScheduler:
    def __init__(self, lr, step_size, gamma=0.1):
        """
        Step Learning Rate Scheduler.
        
        Args:
            lr (float): Initial learning rate.
            step_size (int): Number of steps between learning rate reductions.
            gamma (float): Multiplicative factor for learning rate decay (default: 0.1).
        """
        self.initial_lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.current_step = 0
    
    def get_lr(self):
        """
        Compute the learning rate for the current step.
        
        Returns:
            float: Current learning rate.
        """
        step_count = self.current_step // self.step_size
        lr = self.initial_lr * (self.gamma ** step_count)
        return lr
    
    def step(self):
        """
        Increment the step counter.
        """
        self.current_step += 1

class MultiStepLRScheduler:
    def __init__(self, lr, milestones, gamma=0.1):
        """
        MultiStep Learning Rate Scheduler.
        
        Args:
            lr (float): Initial learning rate.
            milestones (list): List of step indices at which to reduce learning rate.
            gamma (float): Multiplicative factor for learning rate decay (default: 0.1).
        """
        self.initial_lr = lr
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_step = 0
    
    def get_lr(self):
        """
        Compute the learning rate for the current step.
        
        Returns:
            float: Current learning rate.
        """
        decay_count = sum(1 for milestone in self.milestones if self.current_step >= milestone)
        lr = self.initial_lr * (self.gamma ** decay_count)
        return lr
    
    def step(self):
        """
        Increment the step counter.
        """
        self.current_step += 1