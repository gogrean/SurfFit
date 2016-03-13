import pyfits
import numpy as np
from load_data import Image, load_region
import matplotlib.pyplot as plt
from models import Beta, Constant
#from profile import Box #, counts_profile

import warnings
#warnings.simplefilter('error')

root = "./"
# src_img = Image(root + "mos2S002-obj-im-500-4000.fits")
# bkg_img = Image(root + "mos2S002-back-im-sky-500-4000.fits")
# exp_img = Image(root + "mos2S002-exp-im-500-4000-nosrc.fits")

src_img = Image(root + "comb-obj-im-500-4000.fits")
bkg_img = Image(root + "comb-back-im-sky-500-4000.fits")
exp_img = Image(root + "comb-exp-im-500-4000-nosrc.fits")

region = load_region(root + "relic-sector.reg")

pyfits.writeto('test.fits', src_img.data, clobber=True)

 # Plot counts profile.
p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=25)

mod = Beta()
mod.set_parameter('beta', 0.8)
mod.set_parameter('rc', 2.0)
mod.set_parameter('s0', 1e-2)
mod.set_parameter('const', 1e-3)
print(mod.fit(p, statistics='cash', min_range=0.4, max_range=8.3))
region.plot_profile(p, xlog=True, ylog=True, \
    model_name="beta", model=mod, \
    xlabel=r"Distance (arcmin)", \
    ylabel=r"photons s$^{-1}$ cm$^{-2}$ pixel$^{-1}$")
#
#
# mod = Model("beta") + Model("const")
# params = mod.set_params(beta=0.7, rc=0.4, s0=1e-4, const=3e-6)
#
# mod = Beta()
# mod.show_params()

# CONVERGENCE WITH THESE CONSTRAINTS FOR CHI STATS
# mod.set_parameter('s0', 1e-4)
# mod.set_parameter('beta', 0.8, max_bound = 2.)
# mod.set_parameter('rc', 0.4)

# BUT THE FITTING FAILS WHEN REASONABLE UPPER BOUNDS ARE SET
# mod.set_parameter('s0', 1e-4, max_bound = 1e-2)
# mod.set_parameter('beta', 0.8, max_bound = 2.)
# mod.set_parameter('rc', 0.4)

# CONVERGENCE WITH THESE CONSTRAINTS FOR CHI STATS
# mod.set_parameter('beta', 0.8)
# mod.set_parameter('rc', 0.4)
# mod.set_parameter('s0', 1e-4, max_bound = 1e-2)

# mod.set_constraints([{'type': 'ineq', 'fun': lambda p: p.s0},
#                     {'type': 'ineq', 'fun': lambda p: 3. - p.beta},
#                     {'type': 'ineq', 'fun': lambda p: p.rc}])

# print(mod.fit(p, statistics='cash'))