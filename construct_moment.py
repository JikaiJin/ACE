import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

def kstat_pp(means, r):
    kstat_list = [1]
    for s in range(1, r+1):
        kstat = means[s] - sum([special.binom(s-1, k-1) * means[s-k] * kstat_list[k] for k in range(1, s)])
        kstat_list.append(kstat)
    return kstat_list

def kstat(data, r):
    kstat_list = [1]
    for s in range(1, r+1):
        kstat = np.mean(data ** s)  \
        - sum([special.binom(s-1, k-1) * np.mean(data ** (s-k)) * kstat_list[k] for k in range(1, s)])
        kstat_list.append(kstat)
    return kstat_list

# construct cumulant estimators
def cumulant_estimate(r, res):
    return kstat(res, r)

def multiplier_coefficients(cum_list):
    r = len(cum_list) - 1
    cf_list = [1]
    for s in range(1, r+1):
        cf = - cum_list[s] - sum([special.binom(s-1, k-1) * cf_list[s-k] * cum_list[k] for k in range(1, s)])
        cf_list.append(cf)
    for s in range(1, r+1):
        cf_list[s] *= special.binom(r, s)
    return cf_list

# construct polynomial multipliers up to order r
def multipliers(r, cf_list, eta):
    return np.sum([cf_list[i] * eta**(r-i) for i in range(r+1)], axis = 0)