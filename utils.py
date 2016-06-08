import numpy as np
from SurfMessages import InfoMessages
import load_data

def clean_header(hdr):
    """Remove unwanted keywords from the image header.

    Deletes from the image header unncessary keywords such as HISTORY and
    COMMENT, as well as keywords associated with a 3rd and 4th dimension
    (e.g. NAXIS3, NAXIS4). Some radio images are 4D, but the 3rd and 4th
    dimensions are not necessary for plotting the brightness and may
    occasionally cause problems.
    """
    forbidden_keywords = {'HISTORY', 'COMMENT', 'NAXIS3', 'NAXIS4',
        'CTYPE3', 'CTYPE4', 'CRVAL3', 'CRVAL4', 'CDELT3', 'CDELT4',
        'CRPIX3', 'CRPIX4', 'CUNIT3', 'CUNIT4'}
    existing_keywords = [key for key in forbidden_keywords if key in hdr]
    if any(existing_keywords):
        for key in existing_keywords:
            del hdr[key]
    return hdr

def rotate_point(x0, y0, x, y, angle):
    """Rotate point (x,y) counter-clockwise around (x0,y0)."""
    x_rot = x0 + x * np.cos(angle) - y * np.sin(angle)
    y_rot = y0 + x * np.sin(angle) + y * np.cos(angle)
    return (x_rot, y_rot)

def bin_pix2arcmin(bin_values, pix2arcmin):
    """Convert rates and associated uncertainties from per pixel to
    per arcmin**2 units, and calculate exposure time equivalents for
    source and background observations."""
    bin_radius, bin_width, raw_cts, net_cts, \
        bkg_cts, raw_rate, err_raw_rate, \
        net_rate, err_net_rate, bkg_rate, err_bkg_rate = bin_values
    bin_radius, bin_width = [i * pix2arcmin for i in [bin_radius, bin_width]]
    raw_rate, net_rate, bkg_rate, err_raw_rate, err_net_rate, err_bkg_rate = \
        [i / pix2arcmin**2 for i in [raw_rate, net_rate, bkg_rate,
        err_raw_rate, err_net_rate, err_bkg_rate]]
    t_raw = raw_cts / raw_rate
    if bkg_rate > 0:
        t_bkg = bkg_cts / bkg_rate
    else:
        t_bkg = 0.
    return (bin_radius, bin_width, raw_cts, net_cts,  
            bkg_cts, raw_rate, err_raw_rate, net_rate, err_net_rate, 
            bkg_rate, err_bkg_rate, t_raw, t_bkg)

def get_bkg_exp(counts_img, bkg_img, exp_img):
    if isinstance(counts_img.data, list):
        n_img = len(counts_img.data)
        if not bkg_img:
            bkg_img = []
            for i in range(n_img):
                bkg_img.append(np.zeros_like(counts_img.data[i]))
        elif isinstance(bkg_img, load_data.Image):
            pass
        elif isinstance(bkg_img, list):
            if n_img != len(bkg_img):
                raise TypeError('List of background images should have the \
                    same length as the list of source images.')
            else:
                for i in range(n_img):
                    if not bkg_img[i]:
                        bkg_img[i] = np.zeros_like(counts_img.data[i])
        else:
            raise TypeError('Unrecognized background image format.')
    else:
        if not bkg_img:
            bkg_img = np.zeros_like(counts_img.data)
        elif isinstance(bkg_img, load_data.Image):
            pass
        else:
            raise TypeError('Unrecognized background image format.')

    if isinstance(counts_img.data, list):
        n_img = len(counts_img.data)
        if not exp_img:
            exp_img = []
            for i in range(n_img):
                exp_img.append(np.zeros_like(counts_img.data[i]))
        elif isinstance(exp_img, load_data.Image):
            pass
        elif isinstance(exp_img, list):
            if n_img != len(exp_img):
                raise TypeError('List of exposure images should have the \
                    same length as the list of source images.')
            else:
                for i in range(n_img):
                    if not exp_img[i]:
                        exp_img[i] = np.zeros_like(counts_img.data[i])
        else:
            raise TypeError('Unrecognized exposure image format.')
    else:
        if not exp_img:
            exp_img = np.zeros_like(counts_img.data)
        elif isinstance(exp_img, load_data.Image):
            pass
        else:
            raise TypeError('Unrecognized exposure image format.')
    return bkg_img, exp_img

def merge_subpixel_bins(edges):
    new_edges = [edges[0]]
    start_edge = edges[0]
    for edge in edges[1:]:
        if edge - start_edge >= 1:
            new_edges.append(edge)
            start_edge = edge
    new_edges[-1] = edges[-1]
    return new_edges

def get_edges(max_r, islog):
    if not islog:
        nbins = np.round(max_r)
        # Below, nbins+1 is used because the code gets edges, not
        # bin centers. For nbins there will be nbins+1 edges
        return list(np.linspace(0., max_r, nbins + 1))
    else:
        # If the bins are simply distributed logarithmically such that the
        # smallest bin has a width of at least 1 pixel, one ends up with an
        # unreasonably small number of bins. So, instead, the radial range
        # is divided into many logarithmically-scaled bins, some with
        # sub-pixel width, and then the merging will take care of them
        # later. The end result is a compromise between log-scaling and
        # getting one's time worth of bins.
        nbins = 100.
        min_r = 1.   # to avoid log(0)
        # Below, nbins+1 is used because the code gets edges, not
        # bin centers. For nbins there will be nbins+1 edges.
        edges = np.logspace(np.log10(min_r), np.log10(max_r), nbins + 1)
        # Inserts the 0 edge back into the array of edges.
        edges = list(np.insert(edges, 0, 0.))
        return merge_subpixel_bins(edges)

def get_data_for_chi(profile, minrange, maxrange):
    nbins = len(profile)
    r = np.array([profile[i][0] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    w = np.array([profile[i][1] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    net = np.array([profile[i][7] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    net_err = np.array([profile[i][8] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    return nbins, r, w, net, net_err

def get_data_for_cash(profile, minrange, maxrange):
    nbins = len(profile)
    r = np.array([profile[i][0] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    w = np.array([profile[i][1] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    raw_cts = np.array([profile[i][2] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    bkg = np.array([profile[i][5] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    sb_to_counts_factor = np.array([profile[i][9] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    return nbins, r, w, raw_cts, bkg, sb_to_counts_factor

def call_model(func_name):
    model = getattr(model_defs, func_name)
    if not model:
        raise Exception("Model %s is not implemented." % func_name)
    else:
        return model
