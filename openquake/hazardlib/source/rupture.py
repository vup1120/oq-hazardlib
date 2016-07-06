# coding: utf-8
# The Hazard Library
# Copyright (C) 2012-2016 GEM Foundation
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
Module :mod:`openquake.hazardlib.source.rupture` defines classes
:class:`Rupture`, :class:`BaseProbabilisticRupture` and its subclasses
:class:`NonParametricProbabilisticRupture` and
:class:`ParametricProbabilisticRupture`
"""
import abc
import numpy
import math
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.baselib.slots import with_slots
from openquake.hazardlib.geo.mesh import RectangularMesh
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.geodetic import geodetic_distance, distance
from openquake.hazardlib.near_fault import (get_plane_equation, projection_pp,
                                            directp, average_s_rad,
                                            isochone_ratio, get_xyz_from_ll,
                                            vectors2angle)
from openquake.baselib.python3compat import with_metaclass
from openquake.hazardlib.bayless2013model import *


@with_slots
class Rupture(object):
    """
    Rupture object represents a single earthquake rupture.

    :param mag:
        Magnitude of the rupture.
    :param rake:
        Rake value of the rupture.
        See :class:`~openquake.hazardlib.geo.nodalplane.NodalPlane`.
    :param tectonic_region_type:
        Rupture's tectonic regime. One of constants
        in :class:`openquake.hazardlib.const.TRT`.
    :param hypocenter:
        A :class:`~openquake.hazardlib.geo.point.Point`, rupture's hypocenter.
    :param surface:
        An instance of subclass of
        :class:`~openquake.hazardlib.geo.surface.base.BaseSurface`.
        Object representing the rupture surface geometry.
    :param source_typology:
        Subclass of :class:`~openquake.hazardlib.source.base.BaseSeismicSource`
        (class object, not an instance) referencing the typology
        of the source that produced this rupture.
    :param rupture_slip_direction:
        Angle describing rupture propagation direction in decimal degrees.

    :raises ValueError:
        If magnitude value is not positive, hypocenter is above the earth
        surface or tectonic region type is unknown.

    NB: if you want to convert the rupture into XML, you should set the
    attribute surface_nodes to an appropriate value.
    """
    _slots_ = '''mag rake tectonic_region_type hypocenter surface
    surface_nodes source_typology rupture_slip_direction'''.split()

    def __init__(self, mag, rake, tectonic_region_type, hypocenter,
                 surface, source_typology, rupture_slip_direction=None,
                 surface_nodes=()):
        if not mag > 0:
            raise ValueError('magnitude must be positive')
        if not hypocenter.depth > 0:
            raise ValueError('rupture hypocenter must have positive depth')
        NodalPlane.check_rake(rake)
        self.tectonic_region_type = tectonic_region_type
        self.rake = rake
        self.mag = mag
        self.hypocenter = hypocenter
        self.surface = surface
        self.source_typology = source_typology
        self.surface_nodes = surface_nodes
        self.rupture_slip_direction = rupture_slip_direction


class BaseProbabilisticRupture(with_metaclass(abc.ABCMeta, Rupture)):
    """
    Base class for a probabilistic rupture, that is a :class:`Rupture`
    associated with a temporal occurrence model defining probability of
    rupture occurrence in a certain time span.
    """

    @abc.abstractmethod
    def get_probability_no_exceedance(self, poes):
        """
        Compute and return the probability that in the time span for which the
        rupture is defined, the rupture itself never generates a ground motion
        value higher than a given level at a given site.

        Such calculation is performed starting from the conditional probability
        that an occurrence of the current rupture is producing a ground motion
        value higher than the level of interest at the site of interest.

        The actual formula used for such calculation depends on the temporal
        occurrence model the rupture is associated with.

        The calculation can be performed for multiple intensity measure levels
        and multiple sites in a vectorized fashion.

        :param poes:
            2D numpy array containing conditional probabilities the the a
            rupture occurrence causes a ground shaking value exceeding a
            ground motion level at a site. First dimension represent sites,
            second dimension intensity measure levels. ``poes`` can be obtained
            calling the :meth:`method
            <openquake.hazardlib.gsim.base.GroundShakingIntensityModel.get_poes>`.
        """

    @abc.abstractmethod
    def sample_number_of_occurrences(self):
        """
        Randomly sample number of occurrences from temporal occurrence model
        probability distribution.

        .. note::
            This method is using random numbers. In order to reproduce the
            same results numpy random numbers generator needs to be seeded, see
            http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html

        :returns:
            int, Number of rupture occurrences
        """


class NonParametricProbabilisticRupture(BaseProbabilisticRupture):
    """
    Probabilistic rupture for which the probability distribution for rupture
    occurrence is described through a generic probability mass function.

    :param pmf:
        Instance of :class:`openquake.hazardlib.pmf.PMF`. Values in the
        abscissae represent number of rupture occurrences (in increasing order,
        staring from 0) and values in the ordinates represent associated
        probabilities. Example: if, for a given time span, a rupture has
        probability ``0.8`` to not occurr, ``0.15`` to occur once, and
        ``0.05`` to occur twice, the ``pmf`` can be defined as ::

          pmf = PMF([(Decimal('0.8'), 0), (Decimal('0.15'), 1),
                      Decimal('0.05', 2)])

    :raises ValueError:
        If number of ruptures in ``pmf`` do not start from 0, are not defined
        in increasing order, and if they are not defined with unit step
    """

    def __init__(self, mag, rake, tectonic_region_type, hypocenter, surface,
                 source_typology, pmf, rupture_slip_direction=None):
        x = numpy.array([x for (y, x) in pmf.data])
        if not x[0] == 0:
            raise ValueError('minimum number of ruptures must be zero')
        if not numpy.all(numpy.sort(x) == x):
            raise ValueError(
                'numbers of ruptures must be defined in increasing order')
        if not numpy.all(numpy.diff(x) == 1):
            raise ValueError(
                'numbers of ruptures must be defined with unit step')
        super(NonParametricProbabilisticRupture, self).__init__(
            mag, rake, tectonic_region_type, hypocenter, surface,
            source_typology, rupture_slip_direction
        )
        self.pmf = pmf

    def get_probability_no_exceedance(self, poes):
        """
        See :meth:`superclass method
        <.rupture.BaseProbabilisticRupture.get_probability_no_exceedance>`
        for spec of input and result values.

        Uses the formula ::

            ∑ p(k|T) * p(X<x|rup)^k

        where ``p(k|T)`` is the probability that the rupture occurs k times in
        the time span ``T``, ``p(X<x|rup)`` is the probability that a rupture
        occurrence does not cause a ground motion exceedance, and the summation
        ``∑`` is done over the number of occurrences ``k``.

        ``p(k|T)`` is given by the constructor's parameter ``pmf``, and
        ``p(X<x|rup)`` is computed as ``1 - poes``.
        """
        p_kT = numpy.array([float(p) for (p, _) in self.pmf.data])
        prob_no_exceed = numpy.array(
            [v * ((1 - poes) ** i) for i, v in enumerate(p_kT)]
        )
        prob_no_exceed = numpy.sum(prob_no_exceed, axis=0)

        return prob_no_exceed

    def sample_number_of_occurrences(self):
        """
        See :meth:`superclass method
        <.rupture.BaseProbabilisticRupture.sample_number_of_occurrences>`
        for spec of input and result values.

        Uses 'Inverse Transform Sampling' method.
        """
        # compute cdf from pmf
        cdf = numpy.cumsum([float(p) for p, _ in self.pmf.data])

        rn = numpy.random.random()
        [n_occ] = numpy.digitize([rn], cdf)

        return n_occ


@with_slots
class ParametricProbabilisticRupture(BaseProbabilisticRupture):
    """
    :class:`Rupture` associated with an occurrence rate and a temporal
    occurrence model.

    :param occurrence_rate:
        Number of times rupture happens per year.
    :param temporal_occurrence_model:
        Temporal occurrence model assigned for this rupture. Should
        be an instance of :class:`openquake.hazardlib.tom.PoissonTOM`.

    :raises ValueError:
        If occurrence rate is not positive.
    """
    _slots_ = Rupture._slots_ + [
        'occurrence_rate', 'temporal_occurrence_model']

    def __init__(self, mag, rake, tectonic_region_type, hypocenter, surface,
                 source_typology, occurrence_rate, temporal_occurrence_model,
                 rupture_slip_direction=None):
        if not occurrence_rate > 0:
            raise ValueError('occurrence rate must be positive')
        super(ParametricProbabilisticRupture, self).__init__(
            mag, rake, tectonic_region_type, hypocenter, surface,
            source_typology, rupture_slip_direction
        )
        self.temporal_occurrence_model = temporal_occurrence_model
        self.occurrence_rate = occurrence_rate

    def get_probability_one_or_more_occurrences(self):
        """
        Return the probability of this rupture to occur one or more times.

        Uses
        :meth:`~openquake.hazardlib.tom.PoissonTOM.get_probability_one_or_more_occurrences`
        of an assigned temporal occurrence model.
        """
        tom = self.temporal_occurrence_model
        rate = self.occurrence_rate
        return tom.get_probability_one_or_more_occurrences(rate)

    def get_probability_one_occurrence(self):
        """
        Return the probability of this rupture to occur exactly one time.

        Uses :meth:
        `~openquake.hazardlib.tom.PoissonTOM.get_probability_one_occurrence`
        of an assigned temporal occurrence model.
        """
        tom = self.temporal_occurrence_model
        rate = self.occurrence_rate
        return tom.get_probability_one_occurrence(rate)

    def sample_number_of_occurrences(self):
        """
        Draw a random sample from the distribution and return a number
        of events to occur.

        Uses :meth:
        `~openquake.hazardlib.tom.PoissonTOM.sample_number_of_occurrences`
        of an assigned temporal occurrence model.
        """
        return self.temporal_occurrence_model.sample_number_of_occurrences(
            self.occurrence_rate
        )

    def get_probability_no_exceedance(self, poes):
        """
        See :meth:`superclass method
        <.rupture.BaseProbabilisticRupture.get_probability_no_exceedance>`
        for spec of input and result values.

        Uses
        :meth:`~openquake.hazardlib.tom.PoissonTOM.get_probability_no_exceedance`
        """
        tom = self.temporal_occurrence_model
        rate = self.occurrence_rate
        return tom.get_probability_no_exceedance(rate, poes)

    def get_dppvalue(self, site):
        """
        Get the directivity prediction value, DPP at
        a given site as described in Spudich et al. (2013).

        :param site:
            :class:`~openquake.hazardlib.geo.point.Point` object
            representing the location of the target site
        :returns:
            A float number, directivity prediction value (DPP).
        """

        origin = self.surface.get_resampled_top_edge()[0]
        dpp_multi = []
        index_patch = self.surface.hypocentre_patch_index(
            self.hypocenter, self.surface.get_resampled_top_edge(),
            self.surface.mesh.depths[0][0], self.surface.mesh.depths[-1][0],
            self.surface.get_dip())
        idx_nxtp = True
        hypocenter = self.hypocenter

        while idx_nxtp:

            # E Plane Calculation
            p0, p1, p2, p3 = self.surface.get_fault_patch_vertices(
                self.surface.get_resampled_top_edge(),
                self.surface.mesh.depths[0][0],
                self.surface.mesh.depths[-1][0],
                self.surface.get_dip(), index_patch=index_patch)

            [normal, dist_to_plane] = get_plane_equation(
                p0, p1, p2, origin)

            pp = projection_pp(site, normal, dist_to_plane, origin)
            pd, e, idx_nxtp = directp(
                p0, p1, p2, p3, hypocenter, origin, pp)
            pd_geo = origin.point_at(
                (pd[0] ** 2 + pd[1] ** 2) ** 0.5, -pd[2],
                numpy.degrees(math.atan2(pd[0], pd[1])))

            # determine the lower bound of E path value
            f1 = geodetic_distance(p0.longitude,
                                   p0.latitude,
                                   p1.longitude,
                                   p1.latitude)
            f2 = geodetic_distance(p2.longitude,
                                   p2.latitude,
                                   p3.longitude,
                                   p3.latitude)

            if f1 > f2:
                f = f1
            else:
                f = f2

            fs, rd, r_hyp = average_s_rad(site, hypocenter, origin,
                                          pp, normal, dist_to_plane, e, p0,
                                          p1, self.rupture_slip_direction)
            cprime = isochone_ratio(e, rd, r_hyp)

            dpp_exp = cprime * numpy.maximum(e, 0.1 * f) *\
                numpy.maximum(fs, 0.2)
            dpp_multi.append(dpp_exp)

            # check if go through the next patch of the fault
            index_patch = index_patch + 1

            if (len(self.surface.get_resampled_top_edge())
                <= 2) and (index_patch >=
                           len(self.surface.get_resampled_top_edge())):

                idx_nxtp = False
            elif index_patch >= len(self.surface.get_resampled_top_edge()):
                idx_nxtp = False
            elif idx_nxtp:
                hypocenter = pd_geo
                idx_nxtp = True

        # calculate DPP value of the site.
        if numpy.sum(dpp_multi) > 0.:
            dpp = numpy.log(numpy.sum(dpp_multi))
        else:
            dpp = numpy.log(0.8 * 0.1 * f * 0.2)
        return dpp

    def get_cdppvalue(self, target, buf=1.0, delta=0.01, space=2.):
        """
        Get the directivity prediction value, centred DPP(cdpp) at
        a given site as described in Spudich et al. (2013), and this cdpp is
        used in Chiou and Young(2014) GMPE for near-fault directivity
        term prediction.

        :param target_site:
            A mesh object representing the location of the target sites.
        :param buf:
            A float value presents  the buffer distance in km to extend the
            mesh borders to.
        :param delta:
            A float value presents the desired distance between two adjacent
            points in mesh
        :param space:
            A float value presents the tolerance for the same distance of the
            sites (default 2 km)
        :returns:
            A float value presents the centreed directivity predication value
            which used in Chioud and Young(2014) GMPE for directivity term
        """

        min_lon, max_lon, max_lat, min_lat = self.surface.get_bounding_box()

        min_lon -= buf
        max_lon += buf
        min_lat -= buf
        max_lat += buf

        lons = numpy.arange(min_lon, max_lon + delta, delta)
        lats = numpy.arange(min_lat, max_lat + delta, delta)
        lons, lats = numpy.meshgrid(lons, lats)

        target_rup = self.surface.get_min_distance(target)
        mesh = RectangularMesh(lons=lons, lats=lats, depths=None)
        mesh_rup = self.surface.get_min_distance(mesh)

        target_lons = target.lons
        target_lats = target.lats
        cdpp = numpy.empty(len(target_lons))

        for iloc, (target_lon, target_lat) in enumerate(zip(target_lons,
                                                            target_lats)):
            if target_rup[iloc] <= 70.:
                cdpp_sites_lats = mesh.lats[(mesh_rup <= target_rup[iloc] + space)
                                            & (mesh_rup >= target_rup[iloc]
                                               - space)]
                cdpp_sites_lons = mesh.lons[(mesh_rup <= target_rup[iloc] + space)
                                            & (mesh_rup >= target_rup[iloc]
                                               - space)]

                dpp_sum = []
                dpp_target = self.get_dppvalue(Point(target_lon, target_lat))

                for lon, lat in zip(cdpp_sites_lons, cdpp_sites_lats):
                    site = Point(lon, lat, 0.)
                    dpp_one = self.get_dppvalue(site)
                    dpp_sum.append(dpp_one)

                mean_dpp = numpy.mean(dpp_sum)
                cdpp[iloc] = dpp_target - mean_dpp
            else:
                cdpp[iloc] = 0.

        return cdpp

    def get_somerviller_rupture_parameters(self, target, output=1):
        """
        Obtain the distance parameters needed to predict directivity for
        strike-slip event defined by Somerville et al., 1997, page 205.

        :param target:
            A mesh object representing the location of the target sites.
        :param angle:
            If ``True`` (by default), the rup_azimuth is calculated. If this
            is set to ``False``, the rup_distance is calculated.
        :param output:
            1: s
            2: theta
            3: d
            4: az
        :returns:
            s, a numpy array, represents the rupture fraction
            distance to the target site for strike-slip
            theta, a numpy array, represents the angle between the
            rupture direction and the path to the site with respect to the
            rupture (measured in degrees herein) for strike-slip
            d, a numpy array, represents the rupture fraction
            distance to the target site for dip-slip
            az, a numpy array, represents the angle between the
            rupture direction and the path to the site with respect to the
            rupture (measured in degrees herein) for dip-slip
        """

        fd = 0.
        for i in range(1, len(self.surface.get_resampled_top_edge())):
            # Read the vertices of each segment
            P0, P1, P2, P3 = self.surface.get_fault_patch_vertices(
                self.surface.get_resampled_top_edge(),
                self.surface.mesh.depths[0][0],
                self.surface.mesh.depths[-1][0],
                self.surface.get_dip(), index_patch=i)
                    # Set up pseudo-hypocenter
            phyp = setPseudoHypo(i, self.surface, self.hypocenter)
            # Currently assuming that the rake is the same on all subfaults.
            SlipCategory = getSlipCategory(self.rake)
            T_Mw = Magnitude_taper(self.mag)
            surf = PlanarSurface.from_corner_points(1., P0, P1, P2, P3)
            Rrup = surf.get_min_distance(target)
            Rx = surf.get_rx_distance(target)
            Ry = surf.get_ry0_distance(target)
            weight = surf.get_area() / self.surface.get_area()
            L = distance(P0.longitude, P0.latitude, P0.depth,
                         P1.longitude, P1.latitude, P1.depth)
            W = surf.get_width()

            d = computeD(phyp, P0, P1, P2, P3, target)
            s, theta = computeThetaAndS(phyp, P0, P1, P2, P3, target)
            az = computeAz(Rx, Ry)
            if output == 1:
                return s
            elif output == 2:
                return np.degrees(theta)
            elif output == 3:
                return d
            elif output == 4:
                return np.degrees(az)

    def get_bayless2013fd(self, target, output=1):
        """
        Get the directivity prediction value, prediceted by Bayless and Somerville,
        2013 at a given site

        :param output:
            1: f_geom_SS
            2: tapering_SS * weight
            3: f_geom_DS
            4: tapering_DS * weight
        """
        fd = 0.
        for i in range(1, len(self.surface.get_resampled_top_edge())):
            # Read the vertices of each segment
            P0, P1, P2, P3 = self.surface.get_fault_patch_vertices(
                self.surface.get_resampled_top_edge(),
                self.surface.mesh.depths[0][0],
                self.surface.mesh.depths[-1][0],
                self.surface.get_dip(), index_patch=i)
                    # Set up pseudo-hypocenter
            phyp = setPseudoHypo(i, self.surface, self.hypocenter)
            # Currently assuming that the rake is the same on all subfaults.
            SlipCategory = getSlipCategory(self.rake)
            T_Mw = Magnitude_taper(self.mag)
            surf = PlanarSurface.from_corner_points(1., P0, P1, P2, P3)
            Rrup = surf.get_min_distance(target)
            Rx = surf.get_rx_distance(target)
            Ry = surf.get_ry0_distance(target)
            weight = surf.get_area() / self.surface.get_area()
            L = distance(P0.longitude, P0.latitude, P0.depth,
                         P1.longitude, P1.latitude, P1.depth)
            W = surf.get_width()

            d = computeD(phyp, P0, P1, P2, P3, target)
            s, theta = computeThetaAndS(phyp, P0, P1, P2, P3, target)
            az = computeAz(Rx, Ry)
            f_geom_SS, tapering_SS = computeSS(s, theta, target, L, T_Mw, Rrup)
            f_geom_DS, tapering_DS = computeDS(d, az, T_Mw, Rx, Rrup, W, target)

            if output == 1:
                return f_geom_SS
            elif output == 2:
                return tapering_SS * weight
            elif output == 3:
                return f_geom_DS
            elif output == 4:
                return tapering_DS * weight
