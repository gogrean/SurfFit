import numpy as np
import matplotlib.pyplot as plt
from utils import rotate_point, bin_pix2arcmin, get_bkg_exp
from SurfMessages import InfoMessages, ErrorMessages
import load_data

class Region(object):

    def get_bin_vals(self, counts_img, bkg_img,
        exp_img, pixels_in_bin, only_net_cts=False):
        """Calculate the number of counts in a bin."""

        if not isinstance(counts_img.data, list):
            bkg_corr = [None]
            counts_img_data, counts_img_hdr = [], []
            counts_img_data.append(counts_img.data)
            counts_img_hdr.append(counts_img.hdr)

            bkg_img_data, bkg_img_hdr = [], []
            exp_img_data = []

            if isinstance(bkg_img, load_data.Image):
                if isinstance(bkg_img.data, list):
                    raise TypeError('If the counts image is a single map, then \
                              the background image cannot be a list of maps.')
                else:
                    bkg_img_data.append(bkg_img.data)
                    bkg_img_hdr.append(bkg_img.hdr)
            else:
                bkg_img_data.append(bkg_img)
                bkg_corr = [1.]

            if isinstance(exp_img, load_data.Image):
                if isinstance(exp_img.data, list):
                    raise TypeError('If the counts image is a single map, then \
                              the exposure image cannot be a list of maps.')
                else:
                    exp_img_data.append(exp_img.data)
            else:
                exp_img_data.append(exp_img)

        else:
            bkg_corr = [None] * len(counts_img.data)
            counts_img_data = counts_img.data
            counts_img_hdr = counts_img.hdr

            if isinstance(exp_img, load_data.Image):
                if not isinstance(exp_img.data, list):
                    exp_img_data = exp_img.data * len(counts_img.data)
                elif len(exp_img.data) != len(counts_img.data):
                    raise ValueError('Exposure map must be either a single \
                        image, or a list of images with the same length as \
                        the length of the list of source images.')
                else:
                    exp_img_data = exp_img.data
            else:
                exp_img_data = exp_img

            if isinstance(bkg_img, load_data.Image):
                if not isinstance(bkg_img.data, list):
                    bkg_img_data = bkg_img.data * len(counts_img.data)
                    bkg_img_hdr = bkg_img.hdr * len(counts_img.data)
                elif len(bkg_img.data) != len(counts_img.data):
                    raise ValueError('Background map must be either a single \
                        image, or a list of images with the same length as \
                        the length of the list of source images.')
                else:
                    bkg_img_data = bkg_img.data
                    bkg_img_hdr = bkg_img.hdr
            else:
                bkg_img_data = bkg_img
                bkg_corr = [1.] * len(counts_img_data)

        raw_cts, net_cts, bkg_cts = 0., 0., 0.
        raw_rate, net_rate, bkg_rate = 0., 0., 0.
        err_raw_rate_sq, err_net_rate_sq, err_bkg_rate_sq = 0., 0., 0.

        exp_raw = 0.
        exp_bkg = 0.
        for i in range(len(counts_img_data)):
            if not bkg_corr[i]:
                if not 'BKGNORM' in bkg_img_hdr[i]:
                    bkgnorm = 1.
                else:
                    bkgnorm = bkg_img_hdr[i]['BKGNORM']
                bkg_corr_i = counts_img_hdr[i]['EXPOSURE'] * bkgnorm / \
                    bkg_img_hdr[i]['EXPOSURE']
            else:
                bkg_corr_i = bkg_corr[i]
                bkgnorm = 1.
            for pixel in pixels_in_bin:
                j, k = pixel[0], pixel[1]
                if exp_img_data[i][j, k] == 0:
                    continue
                exp_val = exp_img_data[i][j, k]
                exp_val_bkg = exp_img_data[i][j, k] * \
                              bkg_img_hdr[i]['EXPOSURE'] / \
                              counts_img_hdr[i]['EXPOSURE']
                raw_cts += counts_img_data[i][j, k]
                bkg_cts += bkg_img_data[i][j, k]
                exp_raw += exp_val
                exp_bkg += exp_val_bkg / bkgnorm
            net_cts = raw_cts - bkg_cts * bkg_corr_i
        if only_net_cts:
            return net_cts
        raw_rate = raw_cts / exp_raw
        bkg_rate = bkg_cts / exp_bkg
        net_rate = raw_rate - bkg_rate
        err_raw_rate = np.sqrt(raw_cts) / exp_raw
        err_bkg_rate = np.sqrt(bkg_cts) / exp_bkg
        err_net_rate = np.sqrt(raw_cts / exp_raw**2 + bkg_cts / exp_bkg**2)
        return raw_cts, net_cts, bkg_cts, \
               raw_rate, err_raw_rate, net_rate, err_net_rate, bkg_rate, err_bkg_rate

    def merge_bins(self, counts_img, bkg_img, exp_img,
                   min_counts, islog=True):
        bkg_img, exp_img = get_bkg_exp(counts_img, bkg_img, exp_img)
        edges = self.make_edges(islog)
        pixels_in_bins = self.distribute_pixels(edges)
        nbins = len(edges) - 1
        npix = len(pixels_in_bins)
        bins = []
        start_edge = edges[0]
        pixels_in_current_bin = []
        for i in range(nbins):
            end_edge = edges[i+1]
            pixels_in_current_bin.extend(
                    [(pixels_in_bins[j][0], pixels_in_bins[j][1])
                      for j in range(npix) if pixels_in_bins[j][2] == i])
            net_counts = self.get_bin_vals(counts_img, bkg_img, exp_img,
                             pixels_in_current_bin, only_net_cts=True)
            if net_counts < min_counts:
                if end_edge == edges[-1] and len(bins) != 0:
                    bins[-1][2].extend(pixels_in_current_bin)
                    updated_last_bin = (bins[-1][0], end_edge, bins[-1][2])
                    bins[-1] = updated_last_bin
                elif end_edge == edges[-1] and len(bins) == 0:
                    error_message = ErrorMessages('001')
                    raise ValueError(error_message)
                else:
                    continue
            else:
                bins.append((start_edge, end_edge, pixels_in_current_bin))
                start_edge = end_edge
                pixels_in_current_bin = []
        return bins

    def profile(self, counts_img, bkg_img, exp_img, min_counts=50, islog=True):
        """Generate count profiles.

        The box is divided into bins based on a minimum number of counts or a
        minimum S/N. The box is divided into bins starting from the bottom up,
        where the bottom is defined as the bin starting at the lowest row in
        the nonrotated box. E.g., if the box is rotated by 135 deg and it
        can be divided into three bins, then the bins will be distributed as:

                                 x
                               x   x
                             x       x
                           x    1st    x
                         x   x           x
                       x       x    bin    x
                     x    2nd    x       x
                   x   x           x   x
                 x       x    bin    x
               x    3rd    x       x
                 x           x   x
                   x    bin    x
                     x       x
                       x   x
                         x

        If a background map is not read in, then the background is set to zero.
        If an exposure map is not read in, then the exposure is set to one
        everywhere. However, if point source exclusion is important, then an
        exposure map is necessary; regions with zero exposure will be ignored,
        so they need to be masked in the exposure map. Removing point sources
        from the exposure map with dmcopy can be very slow if the full field of
        view is not included in the region file. So to quicken things up, the
        region file should look something like:

                field()
                -ellipse(......)
                -circle(......)

        The function returns a list of tuples of the form:
        (bin radius, bin width, source counts, source counts uncertainty,
        background counts, background counts uncertainty, net counts,
        net counts uncertainty)
        """
        bkg_img, exp_img = get_bkg_exp(counts_img, bkg_img, exp_img)
        if isinstance(counts_img.hdr, list):
            pix2arcmin = counts_img.hdr[0]['CDELT2'] * 60.
        else:
            pix2arcmin = counts_img.hdr['CDELT2'] * 60.

        bins = self.merge_bins(counts_img, bkg_img, exp_img,
                               min_counts, islog)

        profile = []
        for current_bin in bins:
            raw_cts, net_cts, bkg_cts, \
                raw_rate, err_raw_rate, net_rate, err_net_rate, \
                bkg_rate, err_bkg_rate = \
                    self.get_bin_vals(counts_img, bkg_img, exp_img,
                                      current_bin[2])
            bin_radius = (current_bin[0] + current_bin[1]) / 2.
            bin_width = current_bin[1] - bin_radius
            bin_values = (bin_radius, bin_width, raw_cts, 
                          net_cts, bkg_cts, 
                          raw_rate, err_raw_rate, net_rate, err_net_rate,
                          bkg_rate, err_bkg_rate)
            bin_values = bin_pix2arcmin(bin_values, pix2arcmin)
            profile.append(bin_values)
        return profile

    def counts_profile(self, counts_img, bkg_img, bkg_err_img, exp_img,
        min_counts=100, islog=True):
        """some docstring"""
        return self.profile(counts_img, bkg_img, exp_img,
            min_counts, islog=islog)

    def sb_profile(self, counts_img, bkg_img, exp_img,
        min_counts=100, islog=True):
        """some docstring"""
        return self.profile(counts_img, bkg_img, exp_img,
            min_counts, islog=islog)

    def plot_profile(self, profile, xlog=True, ylog=True,
        xlims=None, ylims=None, xlabel=None, ylabel=None,
        model_name=None, model=None):
        """Plot profile and (optional) fitted model.

        Plots the net count profile (with error bars) and the background
        profile (step function without error bars - the uncertainties are
        difficult to get from a renormalized background image without knowing
        the normalization). The plotting can easily be done without the use of
        this routine, by just calling count_profile to get the data. This would
        allow for more customization than this routine provides.
        """
        nbins = len(profile)

        r = np.array([profile[i][0] for i in range(nbins)])
        r_err = np.array([profile[i][1] for i in range(nbins)])

        bkg = np.array([profile[i][5] for i in range(nbins)])
        bkg_err = np.array([profile[i][6] for i in range(nbins)])
        net_cts = np.array([profile[i][7] for i in range(nbins)])
        err_net_cts = np.array([profile[i][8] for i in range(nbins)])

        plt.scatter(r, net_cts, c="#1e8f1e", alpha=0.85, s=35, marker="s")
        plt.errorbar(r, net_cts, xerr=r_err, yerr=err_net_cts,
                     linestyle="None", color="#1e8f1e")
        plt.step(r, bkg, where="mid", linewidth=2, color='#1f77b4')
        plt.step(r, bkg - bkg_err, where="mid", linewidth=1, color='#1f77b4',
                                   linestyle='--', alpha=0.5)
        plt.step(r, bkg + bkg_err, where="mid", linewidth=1, color='#1f77b4',
                                   linestyle='--', alpha=0.5)

        plt.rc('text', usetex=False)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        plt.grid(True)

        if xlims is not None:
            plt.xlim([xlims[0], xlims[1]])
        else:
            plt.xlim([0.01, np.max(r + r_err)])

        if ylims is not None:
            plt.ylim([ylims[0], ylims[1]])

        if xlog:
            plt.semilogx()

        if ylog:
            plt.semilogy()

        if model is not None and model_name is not None:
            if not model_name:
                raise Exception("No model is defined.")
            else:
                evaluated_model = model.evaluate(r)
                plt.plot(r, evaluated_model, color="#ffa500", linewidth=2, alpha=0.75)

        plt.show()
