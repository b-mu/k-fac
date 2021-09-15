import math

import torch
import torch.optim as optim

from utils.nystrom_utils import (ComputeI, ComputeG)
from torch import einsum, eye, matmul, cumsum
from torch.linalg import inv, svd

class NystromOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.01,
                 momentum=0.9,
                 damping=0.1,
                 weight_decay=0.003,
                 freq=100):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
       
        super(NystromOptimizer, self).__init__(model.parameters(), defaults)
        
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        # self.grad_outputs = {}
        self.IHandler = ComputeI()
        self.GHandler = ComputeG()
        self.model = model
        self._prepare_model()

        self.steps = 0
        self.m_C = {}
        self.m_I = {}
        self.m_G = {}

        self.freq = freq
        self.damping = damping

    def _save_input(self, module, input):
        # storing the optimized input in forward pass
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            II, I = self.IHandler(input[0].data, module)
            self.m_I[module] = II, I

    def _save_grad_output(self, module, grad_input, grad_output):
        # storing the optimized gradients in backward pass
        if self.acc_stats and self.steps % self.freq == 0:
            GG, G = self.GHandler(grad_output[0].data, module)
            self.m_G[module] = GG, G

    def _prepare_model(self):
        count = 0
        print(self.model)
        print('NGD keeps the following modules:')
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        classname = m.__class__.__name__.lower()
        if classname in ['linear', 'conv2d']:
            I = self.m_I[m][1]
            G = self.m_G[m][1]
            n = I.shape[0]

            J = einsum('ni,no->nio', (I, G)).reshape(n, -1)
            p = einsum('np->p', J * J)
            p_ = p / torch.sum(p)
            i = torch.multinomial(p_, num_samples=1)
            Ji = J[:,i].reshape(-1)

            C = einsum('n,np->np', Ji, J)
            C = einsum('np->p', C)
            w = p[i]
            s = math.sqrt(1 / (torch.dot(C, C) + self.damping * w))

            self.m_C[m] = C * s

        else:
            raise NotImplementedError


    def _get_natural_grad(self, m, damping):
        grad = m.weight.grad.data
        classname = m.__class__.__name__.lower()

        if classname in ['linear', 'conv2d']:
            I = self.m_I[m][1]
            G = self.m_G[m][1]
            n = I.shape[0]

            g = grad.reshape(-1)
            C = self.m_C[m]
            v = torch.dot(C, g) * C
            v = v.view_as(grad)

            updates = (grad - v) / damping, None

        return updates


    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip

        # vg_sum = 0

        # for m in self.model.modules():
        #     classname = m.__class__.__name__
        #     if classname in self.known_modules:
        #         v = updates[m]
        #         vg_sum += (v[0] * m.weight.grad.data).sum().item()
        #         if m.bias is not None:
        #             vg_sum += (v[1] * m.bias.grad.data).sum().item()
        #     elif classname in ['BatchNorm1d', 'BatchNorm2d']:
        #         vg_sum += (m.weight.grad.data * m.weight.grad.data).sum().item()
        #         if m.bias is not None:
        #             vg_sum += (m.bias.grad.data * m.bias.grad.data).sum().item()

        # vg_sum = vg_sum * (lr ** 2)

        # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.model.modules():
            if m.__class__.__name__ in ['Linear', 'Conv2d']:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                # m.weight.grad.data.mul_(nu)
                # if v[1] is not None:
                #     m.bias.grad.data.copy_(v[1])
                    # m.bias.grad.data.mul_(nu)
            # elif m.__class__.__name__ in ['BatchNorm1d', 'BatchNorm2d']:
            #     m.weight.grad.data.mul_(nu)
            #     if m.bias is not None:
            #         m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                # print('=== step ===')
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                #         buf.mul_(momentum).add_(d_p)
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1, d_p)
                #     d_p.copy_(buf)

                # if weight_decay != 0 and self.steps >= 10 * self.freq:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)
                # print('d_p:', d_p.shape)
                # print(d_p)

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.freq == 0:
                self._update_inv(m)
            v = self._get_natural_grad(m, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
