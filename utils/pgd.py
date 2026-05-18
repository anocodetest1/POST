from torch.optim.optimizer import required
from torch.optim import Optimizer
import torch


class PGD(Optimizer):
    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)


        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    def __init__(self):
        self.nuclear_norm = None
  
    def prox_l1_on_sigmoid(self, data: torch.Tensor, alpha: float):
        x = data.clone()
        d = x.shape[-1]
        mask = 1.0 - torch.eye(d, device=x.device, dtype=x.dtype)

        for _ in range(3):
            s = torch.sigmoid(x)
            x = data - alpha * mask * (s * (1.0 - s))

        return x

prox_operators = ProxOperators()

