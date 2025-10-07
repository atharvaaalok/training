import torch
from .adamw import AdamW


def get_scheduler(optimizer, config):
    if config.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.gamma,
            patience=config.scheduler_patience,
            mode="min",
        )
    elif config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_T_max
        )
    elif config.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate,
            div_factor=config.div_factor, final_div_factor=config.final_div_factor,
            pct_start=config.pct_start, steps_per_epoch=1, epochs=config.n_epochs)
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")
    return scheduler


class CombinedOptimizer:
    '''
    CombinedOptimizer.
    train two param groups independently.
    the learning rates of two optimizers are lr, lr_ratio*lr respectively
    '''

    def __init__(self, params1, params2, betas, lr, lr_ratio, weight_decay):
        self.optimizer1 = AdamW(
            params1,
            betas=betas,
            lr=lr,
            weight_decay=weight_decay,
        )
        if params2 == []:
            self.optimizer2 = None
        else:
            self.optimizer2 = AdamW(
                params2,
                betas=betas,
                lr=lr_ratio * lr,
                weight_decay=weight_decay,
            )

    @property
    def param_groups(self):
        if self.optimizer2:
            return self.optimizer1.param_groups + self.optimizer2.param_groups
        return self.optimizer1.param_groups

    def step(self):
        self.optimizer1.step()
        if self.optimizer2:
            self.optimizer2.step()

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer1.zero_grad(set_to_none=set_to_none)
        if self.optimizer2:
            self.optimizer2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Initialize an empty dictionary to store the state
        state = {}
        # Save the state of the first optimizer
        state['optimizer1'] = self.optimizer1.state_dict()
        if self.optimizer2:
            # Save the state of the second optimizer
            state['optimizer2'] = self.optimizer2.state_dict()
        return state

    def load_state_dict(self, state_dict):
        # Load the state of the first optimizer
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        if self.optimizer2:
            # Load the state of the second optimizer
            self.optimizer2.load_state_dict(state_dict['optimizer2'])


class OneCycleLRCombinedScheduler:
    '''
    Combinedscheduler.
    scheduler two optimizers independently.
    '''

    def __init__(self, Combinedoptimizer, max_lr, lr_ratio,
                 div_factor, final_div_factor, pct_start,
                 steps_per_epoch, epochs):

        self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            Combinedoptimizer.optimizer1, max_lr=max_lr,
            div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start,
            steps_per_epoch=steps_per_epoch, epochs=epochs)
        if Combinedoptimizer.optimizer2:
            self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
                Combinedoptimizer.optimizer2, max_lr=max_lr * lr_ratio,
                div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start,
                steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            self.scheduler2 = None

    def step(self):
        self.scheduler1.step()
        if self.scheduler2:
            self.scheduler2.step()

    def state_dict(self):
        # Initialize an empty dictionary to store the state
        state = {}

        # Save the state of the first scheduler
        state['scheduler1'] = self.scheduler1.state_dict()
        if self.scheduler2:
            # Save the state of the second scheduler
            state['scheduler2'] = self.scheduler2.state_dict()

        return state

    def load_state_dict(self, state_dict):
        # Load the state of the first scheduler
        self.scheduler1.load_state_dict(state_dict['scheduler1'])
        if self.scheduler2:
            # Load the state of the second scheduler
            self.scheduler2.load_state_dict(state_dict['scheduler2'])
