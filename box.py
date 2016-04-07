from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages, InfoMessages
from utils import rotate_point, get_edges
import prof

class Box(prof.Region):
    """Generate box object."""
    def __init__(self, x0, y0, width, height, angle):
       self.x0 = x0
       self.y0 = y0
       self.width = width
       self.height = height
       self.angle = angle

    @classmethod
    def from_params(cls, params):
        """Make box parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the box center. The rotation angle
        of the box is also converted from degrees (as defined in DS9) to radians.
        """
        angle = params[4] * np.pi / 180.
        return Box(params[0] - 1, params[1] - 1, params[2], params[3], angle)

    def get_corners(self):
        """Get the coordinates of the corners of a box."""
        corners = [(-self.width/2, -self.height/2),
                   (-self.width/2, self.height/2),
                   (self.width/2, self.height/2),
                   (self.width/2, -self.height/2)]
        rotated_corners = [rotate_point(self.x0, self.y0, x, y, self.angle)
            for x, y in corners]
        return rotated_corners

    def make_edges(self, islog):
        return get_edges(self.height, islog)

    def distribute_pixels(self, edges):
        corners = self.get_corners()
        reg_path = Path(corners)
        # Get region boundaries.
        bounds = reg_path.get_extents().get_points()
        [[x_min_bound, y_min_bound], [x_max_bound, y_max_bound]] = bounds
        # For cases when the boundary pixels are not integers:
        x_min_bound = floor(x_min_bound)
        y_min_bound = floor(y_min_bound)
        x_max_bound = ceil(x_max_bound)
        y_max_bound = ceil(y_max_bound)
        pixels_in_bins = []
        for x in range(x_min_bound, x_max_bound+1):
            for y in range(y_min_bound, y_max_bound+1):
                if reg_path.contains_point((x, y)):
                    x_nonrotated, y_nonrotated = rotate_point(self.x0, self.y0,
                                                              x - self.x0,
                                                              y - self.y0,
                                                              -self.angle)
                    dist_from_box_bottom = self.height/2. - \
                                           (self.y0 - y_nonrotated)
                    for i, edge in enumerate(edges[1:]):
                        if edge > dist_from_box_bottom:
                            pixels_in_bins.append((y, x, i))
                            break
        return pixels_in_bins
