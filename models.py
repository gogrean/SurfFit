import numpy as np
#import numpy.polynomial.legendre
import scipy.integrate
from astropy.modeling import Fittable1DModel, Parameter, Model
from scipy import integrate
from datetime import datetime
import bknpow_def
import os

def IntModel(model_cls):
    class MyIntModel(model_cls):
        def __init__(self, widths, order=5, *args, **kwargs):
            self._widths = widths
            self._roots, self._weights = np.polynomial.legendre.leggauss(order)
            super(MyIntModel, self).__init__(*args, **kwargs)

        def __getnewargs__(self):
            return (model_cls,)

        def __getstate__(self):
            print(os.getpid())
            return (model_cls, super(MyIntModel, self).__dict__)

        def __setstate__(self, state):
            cls, attributes = state
            obj = cls.__new__(cls)
            self.__dict__.update(attributes)
            return obj
            #print(os.getpid())
            #print(model_cls, os.getpid())
            #return None

        def evaluate_for_integral(self, fn, a, b, *params):
            return (b - a) / 2.0 * np.array(fn((b - a) / 2.0 * self._roots + (a + b) / 2.0, *params))

        def evaluate(self, x, *params):
            fn = super(MyIntModel, self).evaluate

            # Gauss-Legendre integration
            sample_points = np.array([self.evaluate_for_integral(fn, d - dw, d + dw, *params) / (2.*dw)
                                      for d, dw in zip(x, self._widths)])
            result = np.sum(self._weights * sample_points, axis=1)
            return result

        def fit_deriv(self, x, *params):
            fn = super(MyIntModel, self).fit_deriv

            sample_points = np.array([self.evaluate_for_integral(fn, d - dw, d + dw, *params) / (2.*dw)
                                      for d, dw in zip(x, self._widths)])
            result = np.sum(self._weights * sample_points, axis=2).T
            return result

    return MyIntModel

class Beta(Fittable1DModel):
    s0 = Parameter(default = 1e-2, min = 1e-12)
    beta = Parameter(default = 0.7, min = 1e-12)
    rc = Parameter(default = 0.1, min = 1e-12)
    const = Parameter(default = 1e-3, min=1e-12)

    @staticmethod
    def evaluate(x, s0, beta, rc, const):
        result = s0 * (1. + (x/rc)**2) ** (0.5 - 3*beta) + const
        return result

    @staticmethod
    def fit_deriv(x, s0, beta, rc, const):
        d_s0 = (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_beta = -3 * s0 * np.log(1. + (x/rc)**2) * \
                 (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_rc = -2 * s0 * x**2 * (0.5 - 3*beta) / rc**3 \
               * (1. + (x/rc)**2) ** (-0.5 - 3*beta)
        return [d_s0, d_beta, d_rc, np.ones_like(x)]

class BrokenPow(Fittable1DModel):
    ind1 = Parameter(default = 0.)
    ind2 = Parameter(default = 0.)
    norm = Parameter(default = 1e-2, min = 1e-12)
    rbreak = Parameter(default = 0.1, min = 1e-12)
    jump = Parameter(default = 2.0, min = 1., max = 4.)
    const = Parameter(default = 1e-3)

    """
    @staticmethod
    def evaluate_with_derivatives(...):
        ...
    """

    @staticmethod
    def evaluate(x, ind1, ind2, norm, rbreak, jump, const):
#        print("evaluate bknpow - start time: ", str(datetime.now()))
        if not isinstance(x, (int, float)):
            sx = np.zeros_like(x)
            for i in range(len(sx)):
                sx[i] = BrokenPow.evaluate_one(x[i], ind1, ind2, norm, rbreak, jump)
        else:
            sx = BrokenPow.evaluate_one(x, ind1, ind2, norm, rbreak, jump)
#        print("evaluate bknpow - start time: ", str(datetime.now()))
        return sx+const

    @staticmethod
    def evaluate_one(xval, ind1, ind2, norm, rbreak, jump):
        norm_after_jump = norm / jump**2
        if xval <= rbreak:
            return norm * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind1),
                               1e-4, np.sqrt(rbreak**2 - xval**2))[0] + \
                    norm_after_jump * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind2),
                               np.sqrt(rbreak**2 - xval**2), 1e4)[0]
        else:
            return norm_after_jump * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind2),
                                1e-4, 1e4)[0]

    @staticmethod
    def fit_deriv(x, ind1, ind2, norm, rbreak, jump, const):
        if not isinstance(x, (int, float)):
            d_ind1 = np.zeros_like(x)
            d_ind2 = np.zeros_like(x)
            d_norm = np.zeros_like(x)
            d_rbreak = np.zeros_like(x)
            d_jump = np.zeros_like(x)
            for i in range(len(x)):
                d_ind1_tmp, d_ind2_tmp, d_norm_tmp, \
                    d_rbreak_tmp, d_jump_tmp = BrokenPow.fit_deriv_one(
                    x[i], ind1, ind2, norm, rbreak, jump)
                d_ind1[i] = d_ind1_tmp
                d_ind2[i] = d_ind2_tmp
                d_norm[i] = d_norm_tmp
                d_rbreak[i] = d_rbreak_tmp
                d_jump[i] = d_jump_tmp
        else:
            d_ind1, d_ind2, d_norm, d_rbreak, d_jump = \
                BrokenPow.fit_deriv_one(x, ind1, ind2, norm, rbreak, jump)
        d_const = np.ones_like(x)
        return [d_ind1, d_ind2, d_norm, d_rbreak, d_jump, d_const]

    @staticmethod
    def fit_deriv_one(xval, ind1, ind2, norm, rbreak, jump):
        fn1 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind1)
        fn2 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind2)
        norm_after_jump = norm / jump**2
        if xval <= rbreak:
            lim = (rbreak**2 - xval**2)**0.5
            tmp1 = scipy.integrate.quad(fn1, 1e-4, lim)[0]
            tmp2 = scipy.integrate.quad(fn2, lim, 1e4)[0]
            d_ind1 = -norm * scipy.integrate.quad(lambda z: fn1(z) *
                         np.log((xval**2 + z**2)/rbreak**2), 1e-4, lim)[0]

            d_ind2 = -norm_after_jump * scipy.integrate.quad(lambda z: fn2(z) *
                         np.log((xval**2 + z**2)/rbreak**2), lim, 1e4)[0]

            d_norm = tmp1 + 1./jump**2 * tmp2

            # This is only kind of correct. The derivative of the function
            # with respect to rbreak is not continuous, so the Leibniz rule
            # doesn't apply. But in principle this is should work okay as long
            # as xval != rbreak (which is very unlikely in general).
            d_rbreak = norm * (2. * ind1/rbreak * tmp1 + rbreak / lim) + \
                       norm_after_jump * \
                              (2. * ind2/rbreak * tmp2 - rbreak / lim)

            d_jump = -2 * norm / jump**3 * tmp2

        else:
            tmp = scipy.integrate.quad(fn2, 1e-4, 1e4)[0]
            d_ind1 = np.zeros_like(xval)

            d_ind2 = -norm_after_jump * scipy.integrate.quad(lambda z: fn2(z) *
                         np.log((xval**2 + z**2)/rbreak**2), 1e-4, 1e4)[0]

            d_norm = 1/jump**2 * tmp

            d_rbreak = 2. * norm_after_jump * ind2/rbreak * tmp

            d_jump = -2*norm / jump**3 * tmp

        return [d_ind1, d_ind2, d_norm, d_rbreak, d_jump]
