from .kfac import KFACOptimizer
from .skfac import SKFACOptimizer
from .ekfac import EKFACOptimizer
from .kbfgs import KBFGSOptimizer
from .kbfgsl import KBFGSLOptimizer
from .kbfgsl_2loop import KBFGSL2LOOPOptimizer
from .kbfgsl_mem_eff import KBFGSLMEOptimizer
from .ngd import NGDOptimizer
from .ngd_stream import NGDStreamOptimizer
from .nystrom import NystromOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'skfac':
        return SKFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'kbfgs':
    	return KBFGSOptimizer
    elif name == 'kbfgsl':
    	return KBFGSLOptimizer
    elif name == 'kbfgsl_2loop':
        return KBFGSL2LOOPOptimizer
    elif name == 'kbfgsl_mem_eff':
        return KBFGSLMEOptimizer
    elif name == 'ngd':
        return NGDOptimizer
    elif name == 'ngd_stream':
        return NGDStreamOptimizer
    elif name == 'nystrom':
        return NystromOptimizer
    else:
        raise NotImplementedError