import numpy as np
from collections import OrderedDict


# Performance-Parameter-Efficiency
def ppe(score, r):
    r = r / 100
    return score * np.exp(-np.log10(r+1))

# Performance-Memory-Efficiency
def pme(score, m, ft_mem):
    mr = m / ft_mem
    return score * np.exp(-np.log10(mr+1))
