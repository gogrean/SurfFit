import pyfits
import numpy as np
from box import Box
from epanda import Epanda
import utils

class Image():
    def __init__(self, filename, ext=0):
        """Return a FITS image and the associated header.

        The image is returned as a numpy array. By default, the first HDU is read.
        A different HDU can be specified with the argument 'ext'. The header of the
        image is modified to remove unncessary keywords such as HISTORY and COMMENT,
        as well as keywords associated with a 3rd and 4th dimension (e.g. NAXIS3,
        NAXIS4).
        """
        if not isinstance(filename, list):
            img_hdu = pyfits.open(filename)
            self.data = img_hdu[ext].data
            self.hdr = utils.clean_header(img_hdu[ext].header)
        else:
            img_hdr = []
            img_data = []
            if ext == 0:
                ext = [ext] * len(filename)
            elif len(ext) != len(filename):
                raise IndexError('Length of the extension array must match \
                    number of images.')
            for i in range(len(filename)):
                img_hdu = pyfits.open(filename[i])
                img_data.append(img_hdu[ext[i]].data)
                img_hdr.append(utils.clean_header(img_hdu[ext[i]].header))
            self.data = img_data
            self.hdr = img_hdr

def read_shape(data):
    """Get region shape and parameters.

    Splits the DS9 region definition into a string describing the region shape
    (e.g. box, circle) and a list of floats containing the parameters of the
    region.
    """
    reg_def = data[3].strip()
    shape = reg_def.split(r'(')[0]
    params = [float(i) for i in reg_def.split(r'(')[1].strip(')').split(',')]
    return (shape, params)

def load_region(filename):
    """Read DS9 region file.

    Reads a DS9 region file in which the region is defined in image coordinates.
    The region file should contain a single region. Compound regions are not
    supported currently.
    """
    reg_file = open(filename)
    regions = {'box': Box.from_params,
               'epanda': Epanda.from_epanda_params,
               'panda': Epanda.from_panda_params,
               'circle': Epanda.from_circle_params,
               'ellipse': Epanda.from_ellipse_params}
    data = reg_file.readlines()
    if len(data) != 4 or data[2].strip() != 'image':
        error_message = ErrorMessages('002')
        raise ValueError(error_message)
    else:
        shape, params = read_shape(data)
        region = regions[shape](params)
        print(shape, params)
        return region
