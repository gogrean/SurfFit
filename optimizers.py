import numpy as np
import warnings

from astropy.modeling.optimizers import Optimization
from astropy.utils.exceptions import AstropyUserWarning

DEFAULT_BOUNDS = (-1e12, 1e12)

class Minimize(Optimization):
    """General optimization algorithm based on `scipy.optimize.minimize`.

    The `Minimize` optimizer allows the use of `scipy.optimize.minimize` and
    the associated optimizer methods with `astropy.modeling`.
    """
    supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self, method='Nelder-Mead'):
        from scipy.optimize import minimize
        super(Minimize, self).__init__(minimize)
        method = method.lower()
        self.supported_constraints = ['fixed', 'tied']
        if method in ['l-bfgs-b', 'tnc']:
            self.supported_constraints.append('bounds')
        elif method == 'cobyla':
            self.supported_constraints.extend(['eqcons', 'ineqcons'])
        elif method == 'slsqp':
            self.supported_constraints.extend(['bounds', 'eqcons', 'ineqcons'])
        self.fit_info = {
            'final_func_val': None,
            'numiter': None,
            'exit_mode': None,
            'num_function_calls': None
        }
        self.method = method

    def __call__(self, objfunc, initval, fargs, **kwargs):
        """
        Run the solver.

        Parameters
        ----------
        objfunc : callable
            objection function
        initval : iterable
            initial guess for the parameter values
        fargs : tuple
            other arguments to be passed to the statistic function
        kwargs : dict
            other keyword arguments to be passed to the solver

        """
        kwargs['options'] = {'maxiter': kwargs.pop('maxiter', self._maxiter),
                             'eps': kwargs.pop('eps', self._eps),
                             'ftol': 1e-8,
                             'factr': 1e4,
                             'eps': 1e-8}

        acc = self._acc
        try:
            acc = kwargs['acc']
        except KeyError:
            try:
                acc = kwargs['xtol']
            except KeyError:
                try:
                    acc = kwargs['tol']
                except KeyError:
                    pass

        model = fargs[0]
        pars = [getattr(model, name) for name in model.param_names]

        if 'bounds' in self.supported_constraints:
            bounds = [par.bounds for par in pars if par.fixed is not True and
                      par.tied is False]
            bounds = np.asarray(bounds)
            for i in bounds:
                if i[0] is None:
                    i[0] =  DEFAULT_BOUNDS[0]
                if i[1] is None:
                    i[1] = DEFAULT_BOUNDS[1]
            # older versions of scipy require this array to be float
            kwargs['bounds'] = np.asarray(bounds, dtype=np.float)

        kwargs['constraints'] = ()
        if 'eqcons' in self.supported_constraints:
            if len(model.eqcons) > 0:
                for eq in model.eqcons:
                    kwargs['constaints'] += ({'type': 'eq', 'fun': eq})
            # if equality contstraints are possible, then inequality
            # constraints are too
            if len(model.ineqcons) > 0:
                for ineq in model.ineqcons:
                    kwargs['constraints'] += ({'type': 'ineq', 'fun': ineq})

        result = self.opt_method(objfunc, initval, method=self.method,
                                 args=fargs, tol=acc, **kwargs)

        self.fit_info['final_func_val'] = result['fun']
        self.fit_info['numiter'] = result['nit']
        self.fit_info['exit_mode'] = result['status']
        self.fit_info['message'] = result['message']

        if result['status'] != 0:
            warnings.warn("The fit may be unsuccessful; check "
                          "fit_info['message'] for more information.",
                          AstropyUserWarning)

        return result['x'], self.fit_info
