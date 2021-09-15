import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, matmul
from torch.nn import Unfold

class ComputeI:

    @classmethod
    def compute_cov_a(cls, a, module):
        return cls.__call__(a, module)

    @classmethod
    def __call__(cls, a, module):
        if isinstance(module, nn.Linear):
            II, I = cls.linear(a, module)
            return II, I
        elif isinstance(module, nn.Conv2d):
            II, I = cls.conv2d(a, module)
            return II, I
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            return None

    @staticmethod
    def conv2d(input, module):
        f = Unfold(
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        I = f(input)
        I = einsum('nil->ni', I)
        return None, I

    @staticmethod
    def linear(input, module):
        I = input
        return None, I

class ComputeG:

    @classmethod
    def compute_cov_g(cls, g, module):
        """
        :param g: gradient
        :param module: the corresponding module
        :return:
        """
        return cls.__call__(g, module)

    @classmethod
    def __call__(cls, g, module):
        if isinstance(module, nn.Conv2d):
            GG, G = cls.conv2d(g, module)
            return GG, G
        elif isinstance(module, nn.Linear):
            GG, G = cls.linear(g, module)
            return GG, G
        else:
            return None
        

    @staticmethod
    def conv2d(g, module):
        n = g.shape[0]
        g_out_sc = n * g
        grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
        G = grad_output_viewed
        G = einsum('nol->no', G)
        return None, G

    @staticmethod
    def linear(g, module):
        n = g.shape[0]
        g_out_sc = n * g
        G = g_out_sc
        module.optimized = True
        GG = None
        return GG, G
