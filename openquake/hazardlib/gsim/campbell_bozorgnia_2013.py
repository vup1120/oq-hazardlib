# The Hazard Library
# Copyright (C) 2013-2014, GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`CampbellBozorgnia2013`, and
:class:'CampbellBozorgnia2013Arbitrary'
"""
from __future__ import division

import numpy as np
from math import log, exp
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import SA


class CampbellBozorgnia2013(GMPE):
    """
    Implements GMPE developed by Shrey Kumar Shahi published as "A
    probabilistic framework to include the effects of near-fault
    directivity in seismic hazard assessment" (2013, Doctoral dissertation,
    Stanford University, 102-140). The GMPE is developed based on Kenneth W.
    Campbell and Yousef Bozorgnia (2008) GMPE but explicitly includes the
    near-fault directivity effects. This class implements the model for
    RotD50 component of the elastic spectra. In this implements, for
    near-fault directivity prediction, only the case with unknown pulse
    period and unknown pulse indicator is availiable.
    """

    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types is spectral acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        SA
    ])

    #: Supported intensity measure component is orientation-independent
    #: average horizontal :attr:`~openquake.hazardlib.const.IMC.RotD50`
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    #: Supported standard deviation type is total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters are Vs30, Vs30 type (measured or inferred),
    #: and depth (km) to the 2.5 km/s shear wave velocity layer (z2pt5)
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'z2pt5'))

    #: Required rupture parameters are magnitude, rake, dip, ztor
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'rake', 'dip', 'ztor'))

    #: Required distance measures are Rrup, Rjb, Rs, Rtheta, Rd, and Rphi
    REQUIRES_DISTANCES = set(('rrup', 'rjb', 'rs', 'rtheta', 'rd', 'rphi'))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type
        C = self.COEFFS[imt]

        # compute median pga on rock (vs30=1100), needed for site response
        # term calculation
        # For spectral accelerations at periods between 0.0 and 0.25 s, Sa (T)
        # cannot be less than PGA on soil, therefore if the IMT is in this
        # period range it is necessary to calculate PGA on soil
        if isinstance(imt, SA) and (imt.period > 0.0) and (imt.period < 0.25):
            get_pga_site = True
        else:
            get_pga_site = False
        pga1100, pga_site = self._compute_imt1100(C,
                                                  sites,
                                                  rup,
                                                  dists,
                                                  get_pga_site)

        # Get the median ground motion
        mean = (self._compute_magnitude_term(C, rup.mag) +
                self._compute_distance_term(C, rup, dists) +
                self._compute_style_of_faulting_term(C, rup) +
                self._compute_hanging_wall_term(C, rup, dists) +
                self._compute_shallow_site_response(C, sites, pga1100) +
                self._compute_basin_response_term(C, sites.z2pt5)
                )

        stddevs = self._get_stddevs(C)

        return mean, stddevs

    def _compute_imt1100(self, C, sites, rup, dists, get_pga_site=False):
        """
        Computes the PGA on reference (Vs30 = 1100 m/s) rock.
        """
        # Calculates simple site response term assuming all sites 1100 m/s
        fsite = (C['c10'] + (C['k2'] * C['n'])) * log(1100. / C['k1'])
        # Calculates the PGA on rock
        pga1100 = np.exp(self._compute_magnitude_term(C, rup.mag) +
                         self._compute_distance_term(C, rup, dists) +
                         self._compute_style_of_faulting_term(C, rup) +
                         self._compute_hanging_wall_term(C, rup, dists) +
                         self._compute_basin_response_term(C, sites.z2pt5) +
                         fsite)
        # If PGA at the site is needed then remove factor for rock and
        # re-calculate on correct site condition
        if get_pga_site:
            pga_site = np.exp(np.log(pga1100) - fsite)
            fsite = self._compute_shallow_site_response(C, sites, pga1100)
            pga_site = np.exp(np.log(pga_site) + fsite)
        else:
            pga_site = None
        return pga1100, pga_site

    def _compute_magnitude_term(self, C, mag):
        """
        Returns the magnitude scaling factor (equation (5.5), page 106)
        """
        fmag = C['c0'] + C['c1'] * mag
        if mag <= 5.5:
            return fmag
        elif mag > 6.5:
            return fmag + (C['c2'] * (mag - 5.5)) + (C['c3'] * (mag - 6.5))
        else:
            return fmag + (C['c2'] * (mag - 5.5))

    def _compute_distance_term(self, C, rup, dists):
        """
        Returns the distance scaling factor (equation (5.6), page 106)
        """
        return (C['c4'] + C['c5'] * rup.mag) * \
            np.log(np.sqrt(dists.rrup ** 2. + C['c6'] ** 2.))

    def _compute_style_of_faulting_term(self, C, rup):
        """
        Returns the style of faulting factor, depending on the mechanism (rake)
        and top of rupture depth (equation (5.7 - 5.8), page 107)
        """
        frv, fnm = self._get_fault_type_dummy_variables(rup.rake)

        if frv > 0.:
            # Top of rupture depth term only applies to reverse faults
            if rup.ztor < 1.:
                ffltz = rup.ztor
            else:
                ffltz = 1.
        else:
            ffltz = 0.
        return (C['c7'] * frv * ffltz) + (C['c8'] * fnm)

    def _get_fault_type_dummy_variables(self, rake):
        """
        Returns the coefficients FRV and FNM, describing if the rupture is
        reverse (FRV = 1.0, FNM = 0.0), normal (FRV = 0.0, FNM = 1.0) or
        strike-slip/oblique-slip (FRV = 0.0, FNM = 0.0). Reverse faults are
        classified as those with a rake in the range 30 to 150 degrees. Normal
        faults are classified as having a rake in the range -150 to -30 degrees
        :returns:
            FRV, FNM
        """
        if (rake > 30.0) and (rake < 150.):
            return 1., 0.
        elif (rake > -150.0) and (rake < -30.0):
            return 0., 1.
        else:
            return 0., 0.

    def _compute_hanging_wall_term(self, C, rup, dists):
        """
        Returns the hanging wall scaling term, the product of the scaling
        coefficient and four separate scaling terms for distance, magnitude,
        rupture depth and dip (equations 5.9, page 107). Individual
        scaling terms defined in separate functions
        """
        return (C['c9'] *
                self._get_hanging_wall_distance_term(dists, rup.ztor) *
                self._get_hanging_wall_magnitude_term(rup.mag) *
                self._get_hanging_wall_depth_term(rup.ztor) *
                self._get_hanging_wall_dip_term(rup.dip))

    def _get_hanging_wall_distance_term(self, dists, ztor):
        """
        Returns the hanging wall distance scaling term (equation 5.10, page
        107)
        """
        fhngr = np.ones_like(dists.rjb, dtype=float)
        idx = dists.rjb > 0.
        if ztor < 1.:
            temp_rjb = np.sqrt(dists.rjb[idx] ** 2. + 1.)
            r_max = np.max(np.column_stack([dists.rrup[idx], temp_rjb]),
                           axis=1)
            fhngr[idx] = (r_max - dists.rjb[idx]) / r_max
        else:
            fhngr[idx] = (dists.rrup[idx] - dists.rjb[idx]) / dists.rrup[idx]
        return fhngr

    def _get_hanging_wall_magnitude_term(self, mag):
        """
        Returns the hanging wall magnitude scaling term (equation 5.11, page
        107)
        """
        if mag <= 6.0:
            return 0.
        elif mag >= 6.5:
            return 1.
        else:
            return 2. * (mag - 6.0)

    def _get_hanging_wall_depth_term(self, ztor):
        """
        Returns the hanging wall depth scaling term (equation 5.12, page 107)
        """
        if ztor >= 20.0:
            return 0.
        else:
            return (20. - ztor) / 20.0

    def _get_hanging_wall_dip_term(self, dip):
        """
        Returns the hanging wall dip scaling term (equation 5.13, page 107)
        """
        if dip > 70.0:
            return (90.0 - dip) / 20.0
        else:
            return 1.0

    def _compute_shallow_site_response(self, C, sites, pga1100):
        """
        Returns the shallow site response term (equation 5.14, page 108)
        """
        stiff_factor = C['c10'] + (C['k2'] * C['n'])
        # Initially default all sites to intermediate rock value
        fsite = stiff_factor * np.log(sites.vs30 / C['k1'])
        # Check for soft soil sites
        idx = sites.vs30 < C['k1']
        if np.any(idx):
            pga_scale = np.log(pga1100[idx] +
                               (C['c'] * ((sites.vs30[idx] / C['k1']) **
                                          C['n']))) - np.log(pga1100[idx]
                                                             + C['c'])
            fsite[idx] = C['c10'] * np.log(sites.vs30[idx] / C['k1']) + \
                (C['k2'] * pga_scale)
        # Any very hard rock sites are rendered to the constant amplification
        # factor
        idx = sites.vs30 >= 1100.
        if np.any(idx):
            fsite[idx] = stiff_factor * log(1100. / C['k1'])

        return fsite

    def _compute_basin_response_term(self, C, z2pt5):
        """
        Returns the basin response term (equation 5.15, page 108)
        """
        fsed = np.zeros_like(z2pt5, dtype=float)
        idx = z2pt5 < 1.0
        if np.any(idx):
            fsed[idx] = C['c11'] * (z2pt5[idx] - 1.0)

        idx = z2pt5 > 3.0
        if np.any(idx):
            fsed[idx] = (C['c12'] * C['k3'] * exp(-0.75)) *\
                (1.0 - np.exp(-0.25 * (z2pt5[idx] - 3.0)))
        return fsed

    def _get_stddevs(self, C):
        """
        Returns the standard deviations. Table 5.5, pages 139
        """
        stddevs = []

        stddevs.append(np.array(C['std']))
        return stddevs


    COEFFS = CoeffsTable(sa_damping=5, table="""\
imt c0  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10 c11 c12 std k1  k2  k3  n
0.01    -1.729  0.51    -0.444  -0.378  -2.208  0.178   5.779   0.239   -0.426  0.675   1.017   -0.03   0.292   0.54    865 -1.186  1.839   1.18
0.02    -1.693  0.511   -0.45   -0.377  -2.222  0.18    5.758   0.23    -0.446  0.697   1.062   -0.027  0.282   0.544   865 -1.219  1.84    1.18
0.03    -1.569  0.509   -0.426  -0.396  -2.258  0.179   5.766   0.21    -0.442  0.731   1.125   -0.036  0.31    0.557   908 -1.273  1.841   1.18
0.05    -1.227  0.519   -0.412  -0.441  -2.363  0.188   5.89    0.164   -0.507  0.803   1.248   -0.032  0.342   0.6 1054    -1.346  1.843   1.18
0.075   -0.67   0.527   -0.433  -0.493  -2.546  0.203   7.116   0.129   -0.585  0.819   1.442   -0.077  0.35    0.649   1086    -1.471  1.845   1.18
0.1 -0.327  0.519   -0.439  -0.496  -2.552  0.2 8.033   0.12    -0.51   0.813   1.546   -0.054  0.318   0.66    1032    -1.624  1.847   1.18
0.15    -0.145  0.504   -0.46   -0.453  -2.487  0.188   8.8 0.146   -0.404  0.765   1.789   -0.067  0.424   0.617   878 -1.931  1.852   1.18
0.2 -0.496  0.495   -0.394  -0.454  -2.302  0.173   7.704   0.167   -0.247  0.706   2   -0.008  0.445   0.592   748 -2.188  1.856   1.18
0.25    -0.894  0.511   -0.332  -0.456  -2.219  0.167   6.695   0.23    -0.167  0.642   2.149   0.001   0.345   0.575   654 -2.381  1.861   1.18
0.3 -1.158  0.519   -0.347  -0.377  -2.117  0.162   6.183   0.22    -0.207  0.676   2.273   -0.034  0.292   0.591   587 -2.518  1.865   1.18
0.4 -1.443  0.525   -0.287  -0.361  -2.101  0.165   5.493   0.285   -0.158  0.619   2.408   0.03    0.237   0.609   503 -2.657  1.874   1.18
0.5 -2.534  0.689   -0.461  -0.236  -2.017  0.151   5.024   0.316   -0.126  0.59    2.369   0.006   0.183   0.633   457 -2.669  1.883   1.18
0.75    -4.8    1.025   -0.767  -0.136  -1.985  0.155   4.294   0.354   -0.068  0.647   1.978   0.002   0.101   0.674   410 -2.401  1.906   1.18
1   -6.388  1.237   -0.83   -0.14   -1.99   0.144   4.207   0.395   0.105   0.545   1.467   0.094   0.14    0.684   400 -1.955  1.929   1.18
1.5 -8.705  1.536   -0.953  -0.197  -1.999  0.148   4.219   0.366   0.021   0.527   0.373   0.215   0.085   0.664   400 -1.025  1.974   1.18
2   -9.742  1.617   -0.735  -0.427  -2.015  0.148   4.409   0.358   0.032   0.402   -0.471  0.212   0.034   0.645   400 -0.299  2.019   1.18
3   -10.652 1.606   -0.106  -0.937  -1.857  0.124   4.633   0.264   -0.043  0.129   -0.795  0.148   0.01    0.642   400 0   2.11    1.18
4   -11.323 1.581   0.276   -1.175  -1.821  0.119   4.647   0.152   -0.02   -0.112  -0.783  0.087   0.053   0.636   400 0   2.2 1.18
5   -11.797 1.583   0.532   -1.386  -1.719  0.11    4.723   -0.024  -0.091  -0.23   -0.756  0.01    0.038   0.644   400 0   2.291   1.18
7.5 -12.623 1.548   0.525   -1.123  -1.73   0.126   4.52    -0.094  -0.404  -0.354  -0.669  0.028   0.068   0.666   400 0   2.517   1.18
10  -13.128 1.608   0.166   -0.811  -1.752  0.125   4.165   -0.301  -0.224  -0.383  -0.476  0.195   0.08    0.635   400 0   2.744   1.18
    """)
