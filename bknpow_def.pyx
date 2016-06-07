import numpy as np
import scipy.integrate

def evaluate_one(xval, ind1, ind2, norm, rbreak, jump):
    norm_after_jump = norm / jump**2
    func1 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind1)
    func2 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind2)
    if xval <= rbreak:
        int_lim = (rbreak**2 - xval**2) ** 0.5
        return norm * scipy.integrate.quad(func1, 1e-4, int_lim)[0] + \
               norm_after_jump * scipy.integrate.quad(func2, int_lim, 1e4)[0]
    else:
        return norm_after_jump * scipy.integrate.quad(func2, 1e-4, 1e4)[0]
