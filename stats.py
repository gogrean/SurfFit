import numpy as np
from datetime import datetime

def cstat_deriv(measured_raw_cts, updated_model, measured_bkg_cts,
                      t_raw, t_bkg, x):
    """
    Calculates the derivatives of the C-statistic.
    """
    model_derivs = np.array(updated_model.fit_deriv(x, *updated_model.parameters))
    model_vals = updated_model(x)

    nbins = len(model_vals)
    nparams = len(updated_model.parameters)
    d_cash = np.zeros(nparams)

    # Some of these calculations are used often, so they are done only once
    # here to speed up the code.
    tmp1 = t_raw + t_bkg
    tmp2 = tmp1 * model_vals
    tmp3 = tmp2 - measured_raw_cts - measured_bkg_cts
    tmp4 = tmp2 * measured_bkg_cts
    tmp5 = tmp1 * model_derivs
    tmp6 = tmp5 * measured_bkg_cts
    tmp7 = t_raw * model_derivs
    tmp8 = t_bkg * model_derivs

    d = (tmp3 ** 2 + 4. * tmp4)**0.5
    f = (-tmp3 + d) / (2. * tmp1)

    d_d = (tmp3 ** 2 + 4. * tmp4)**-0.5 * (2. * tmp6 + tmp3 * tmp5)
    d_f = -0.5 * model_derivs + d_d / (2. * tmp1)

    for i in range(nbins):
        if measured_raw_cts[i] == 0 and measured_bkg_cts[i] > 0:
            d_cash += tmp7[:,i]
        elif measured_bkg_cts[i] == 0 and measured_raw_cts[i] > 0:
            if tmp2[i] < measured_raw_cts[i]:
                d_cash -= tmp8[:,i]
            else:
                d_cash += tmp7[:,i] - \
                          measured_raw_cts[i] * 1. / model_vals[i] * \
                          model_derivs[:,i]
        else:
            d_cash += tmp7[:,i] + tmp1[i] * d_f[:,i] - \
                      measured_raw_cts[i] * \
                      1. / (model_vals[i] + f[i]) * \
                      (model_derivs[:,i] + d_f[:,i]) - \
                      measured_bkg_cts[i] * 1. / f[i] * d_f[:,i]
    return 2. * d_cash

def cstat(measured_raw_cts, updated_model, measured_bkg_cts, t_raw, t_bkg, x):
    """
    C-statistic implementation. [1][2]

    This algorithm requires total and background counts, as well as the
    factors that transform counts to rates.

    .. [1] Cash, W. (1979), "Parameter estimation in astronomy through
           application of the likelihood ratio", ApJ, 228, p. 939-947
    .. [2] Wachter, K., Leach, R., Kellogg, E. (1979), "Parameter estimation
           in X-ray astronomy using maximum likelihood", ApJ, 230, p. 274-287
    """
#    start = datetime.now()
    model_vals = updated_model(x)
    #print(updated_model.parameters)

    # Some of these calculations are used often, so they are done only once
    # here to speed up the code.
    tmp1 = t_raw + t_bkg
    tmp2 = tmp1 * model_vals
    tmp3 = tmp2 - measured_raw_cts - measured_bkg_cts
    tmp4 = tmp2 * measured_bkg_cts
    tmp5 = t_raw * model_vals
    tmp6 = t_bkg * model_vals

    d = (tmp3 ** 2 + 4. * tmp4)**0.5
    f = (-tmp3 + d) / (2. * tmp1)

    nbins = len(model_vals)
    cash = 0.
    for i in range(nbins):
        if measured_raw_cts[i] == 0 and measured_bkg_cts[i] > 0:
            cash += tmp5[i] - measured_bkg_cts[i] * np.log(t_bkg[i] / tmp1[i])
        elif measured_bkg_cts[i] == 0 and measured_raw_cts[i] > 0:
            if tmp2[i] < measured_raw_cts[i]:
                cash -= tmp6[i] + measured_raw_cts[i] * np.log(t_raw[i] / tmp1[i])
            else:
                cash += tmp5[i] + measured_raw_cts[i] * \
                        (np.log(measured_raw_cts[i]/tmp5[i]) - 1)
        else:
            cash += tmp5[i] + tmp1[i] * f[i] - \
                    measured_raw_cts[i] * np.log(t_raw[i] * (model_vals[i] + f[i])) \
                    - measured_bkg_cts[i] * np.log(t_bkg[i] * f[i]) - \
                    measured_raw_cts[i] * (1 - np.log(measured_raw_cts[i])) - \
                    measured_bkg_cts[i] * (1 - np.log(measured_bkg_cts[i]))
    #print('cash ==>', 2.*cash)
    return 2. * cash
