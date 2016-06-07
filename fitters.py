import numpy as np
import pickle
from multiprocessing import Pool

from astropy.modeling.fitting import Fitter

from optimizers import Minimize
from stats import cstat
import corner
import emcee
import os.path
from tabulate import tabulate
from tempfile import TemporaryFile

from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.modeling.utils import get_inputs_and_params

def lnprob(mc_params, model, measured_raw_cts, measured_bkg_cts, t_raw, t_bkg, x):
    _fitter_to_model_params(model, mc_params)
    lnp = 0. # TODO: evaluate prior based on bounds
    lnc = cstat(measured_raw_cts, model, measured_bkg_cts, t_raw, t_bkg, x)
    return lnp - lnc

# def lnprob_int(...):
#     model = IntModel(...)
#     return lnprob(...)

class CstatFitter(Fitter):
    """
    Fit a model using the C-statistic. [1][2]

    Parameters
    ----------
    optimizer : class or callable
        one of the classes in optimizers.py or in astropy.modeling.optimizers
        (default: Minimize)

    .. [1] Cash, W. (1979), "Parameter estimation in astronomy through
           application of the likelihood ratio", ApJ, 228, p. 939-947
    .. [2] Wachter, K., Leach, R., Kellogg, E. (1979), "Parameter estimation
           in X-ray astronomy using maximum likelihood", ApJ, 230, p. 274-287
    """
    supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self, optimizer=None):
        if optimizer is None:
            optimizer = Minimize()

        def opt_func(*args, **kwargs):
            return optimizer(*args, **kwargs)

        super(CstatFitter, self).__init__(opt_func, statistic=cstat)

    def __call__(self, model, x, measured_raw_cts, measured_bkg_cts,
                 t_raw, t_bkg, x_err=None, **kwargs):
        if x_err is not None:
            model = IntModel(model.__class__)(x_err, *model.parameters)

        model_copy = _validate_model(model,
                                     self.supported_constraints)
        farg = _convert_input(x, measured_raw_cts)
        farg = (model_copy, measured_bkg_cts, t_raw, t_bkg) + farg
        p0, _ = _model_to_fit_params(model_copy)

        # TODO: Honor estimate_jacobian in kwargs, and/or determine if
        # model supports jacobian, and/or if fitter supports the jac argument.

        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, jac=self.objective_derivative,
            **kwargs)
        _fitter_to_model_params(model_copy, fitparams)

        return model_copy

    def objective_derivative(self, params, model, measured_bkg_cts, t_raw, t_bkg, x, measured_raw_cts):
        _fitter_to_model_params(model, params)
        return cstat_deriv(measured_raw_cts, model, measured_bkg_cts,
                           t_raw, t_bkg, x)

    def mcmc_err(self, model, x, measured_raw_cts, measured_bkg_cts,
                 t_raw, t_bkg, cl=68.27, nruns=500, nwalkers=100, nburn=100,
                 with_corner=True, corner_filename='triangle.pdf',
                 corner_dpi=144, clobber_corner=True, save_chain=False,
                 chain_filename='chain.dat', clobber_chain=True,
                 floatfmt=".3e", tablefmt='orgtbl', **kwargs):
        """Run Markov Chain Monte Carlo for parameter error estimation.

        `model` should be a fitted model as returned by `__call__`.

        Return the Markov Chain as a 3-dimensional array (walker, step, parameter).
        """
        model_copy = _validate_model(model,
                                     self.supported_constraints)
        params, _ = _model_to_fit_params(model_copy)
        ndim = len(params)
        pos = [params + 1e-4 * np.random.randn(ndim) * params
               for i in range(nwalkers)]

        if not os.path.isfile(chain_filename) or clobber_chain:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2,
                                            args=(model_copy, measured_raw_cts, measured_bkg_cts, t_raw, t_bkg, x))
            sampler.run_mcmc(pos, nruns)
            samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

            #samples = np.array(result).reshape((-1, ndim))
            if save_chain:
                with open(chain_filename, 'wb') as f:
                    pickle.dump(samples, f)
        elif os.path.isfile(chain_filename) and not clobber_chain:
            with open(chain_filename, 'rb') as f:
                samples = pickle.load(f)

        par_names = model_copy.param_names
        if with_corner:
            if os.path.isfile(corner_filename) and not clobber_corner:
                raise Exception("Corner plot already exists and clobber_corner=False.")
            else:
                fig = corner.corner(samples, labels=par_names, bins=[20]*ndim)
                fig.savefig(corner_filename, dpi=corner_dpi)
        lim_lower = 50. - cl / 2.
        lim_upper = 50. + cl / 2.

        val_with_errs = [(v[1], v[2]-v[1], v[1]-v[0])
                         for v in zip(*np.percentile(samples,
                                      [lim_lower, 50., lim_upper], axis=0))]
        fit_data = [[par_names[i], val_with_errs[i][0], -val_with_errs[i][1],
                     val_with_errs[i][2]] for i in range(len(val_with_errs))]

        print('\n'*2)
        print('FIT SUMMARY:')
        print('')
        tab_headers = ['Parameter', 'Value',
                       'Lower Uncertainty', 'Upper Uncertainty']
        print(tabulate(fit_data, headers=tab_headers, tablefmt=tablefmt,
                       floatfmt=floatfmt))
        print('\n'*2)

        return fit_data
