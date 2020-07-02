import torch
from torch.optim.optimizer import Optimizer, required


class IntegerMadam(Optimizer):

    def __init__(self, params, base_lr=0.001, lr_factor=10, levels=4096, p_scale=3.0, g_bound=10.0):

        self.p_scale = p_scale
        self.g_bound = g_bound
        defaults = dict(base_lr=base_lr, lr_factor=lr_factor, levels=levels)
        super(IntegerMadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                lr = group['base_lr']
                lr_factor = group['lr_factor']
                levels = group['levels']

                state = self.state[p]
                if len(state) == 0:
                    state['max'] = self.p_scale*(p*p).mean().sqrt().item()
                    state['integer'] = torch.randint_like(p, low=0, high=levels-1)
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                bias_correction = 1 - 0.999 ** state['step']
                state['exp_avg_sq'] = 0.999 * state['exp_avg_sq'] + 0.001 * p.grad.data**2
                
                g_normed = p.grad.data / (state['exp_avg_sq']/bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)
                
                rounded = torch.round(g_normed*lr_factor)*torch.sign(p.data)
                state['integer'] = (state['integer'] + rounded).clamp_(0,levels-1)
                p.data = torch.sign(p.data) * state['max'] * torch.exp(-lr*state['integer'])

        return loss
