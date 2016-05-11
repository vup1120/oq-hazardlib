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
        dpp = numpy.log(numpy.sum(dpp_multi))

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

        return cdpp

    def get_rupture_fraction_strikeslip(self, target, angle=False):
        """
        Obtain the distance parameters needed to predict directivity for
        strike-slip event defined by Somerville et al., 1997, page 205.

        :param target:
            A mesh object representing the location of the target sites.
        :param angle:
            If ``True`` (by default), the rup_azimuth is calculated. If this
            is set to ``False``, the rup_distance is calculated.
        :returns:
            rup_distance, a numpy array, represents the rupture fraction
            distance to the target site.
            rup_azimuth, a numpy array, represents the angle between the
            rupture direction and the path to the site with respect to the
            rupture (measured in degrees herein)
        """
        # check if the rupture is multi-segment
        top_edge = self.surface.get_resampled_top_edge()
        if len(top_edge) > 2:
            raise ValueError(
                'multi-segment rupture calculation not yet implemented')

        idxs = self.surface.mesh.geodetic_min_distance(target, indices=True)
        cls_lon = self.surface.mesh.lons.take(idxs)
        cls_lat = self.surface.mesh.lats.take(idxs)
        s_lon = target.lons
        s_lat = target.lats
        epi = Point(self.hypocenter.longitude, self.hypocenter.latitude)

        # To calculate the effect of directivity, we calculate the rupture
        # fraction length from epicentre(along strike direction as defined in
        # Somerville et al., 1997) to the site, and the rupture angel which
        # is the angle between the fault strike and the path to the site with
        # respect to the rupture. We calculate first the distance between the
        # closest point(site to the rupture) projected onto surface and
        # epicentre. Then, we project the distance segment onto the strike
        # diretion(assumed the rupture direction) to obtain the rupture
        # fraction effective distance to the site.

        # Obtain the strike direction vector(in Cartesian coordinate system)
        # p_pc is one point along strike direction passing through epicentre
        # p_pc_xy is p_pc in Cartesian coordinate system
        # pe_xy is epicentre in Cartesian coordinate system
        # ppc_pe is the vector from epicentre to p_pc
        p_pc = epi.point_at(1., 0., self.surface.get_strike())
        p_pc_xy = get_xyz_from_ll(p_pc, epi)
        pe_xy = get_xyz_from_ll(epi, epi)
        ppc_pe = (numpy.array(p_pc_xy) - numpy.array(pe_xy))

        rup_azimuth = numpy.empty(len(cls_lon))
        rup_distance = numpy.empty(len(cls_lon))
        iloc = 0

        for (lon, lat, slon, slat) in zip(cls_lon, cls_lat, s_lon, s_lat):

            # Obtain the vector from closest point to epicentre.
            # pc_xy is the closest point in Cartesian coordinate system
            # pc_pe is the vector from cloest point to epicentre
            pc_xy = get_xyz_from_ll(Point(lon, lat), epi)
            pc_pe = (numpy.array(pc_xy) - numpy.array(pe_xy))

            phi = vectors2angle(numpy.array(ppc_pe), pc_pe)
            if phi > (math.pi / 2.):
                phi = math.pi - phi

            rup_distance[iloc] = numpy.linalg.norm(pc_pe) * numpy.cos(phi)
            site_xy = get_xyz_from_ll(Point(slon, slat), epi)

            ps_pe = (numpy.array(site_xy) - numpy.array(pe_xy))

            azimuth = vectors2angle(ppc_pe, ps_pe)
            if azimuth > (math.pi / 2.):
                azimuth = math.pi - azimuth
            rup_azimuth[iloc] = numpy.rad2deg(azimuth)
            iloc += 1
        if angle:
            return rup_azimuth
        else:
            return rup_distance

    def get_rupture_fraction_dipslip(self, target, angle=False):
        """
        Obtain the distance parameters needed to predict directivity for
        non-strike-slip event defined by Somerville et al., 1997, page 205.

        :param target:
            A mesh object representing the location of the target sites.
        :param angle:
            If ``True`` (by default), the rup_azimuth is calculated. If this
            is set to ``False``, the rup_distance is calculated.
        :returns:
            rup_distance, a numpy array, represents the rupture fraction
            distance to the target site.
            rup_azimuth, a numpy array, represents the angle between the
            rupture direction and the path to the site with respect to the
            rupture (measured in degrees herein)
        """
        # check if the rupture is multi-patches
        top_edge = self.surface.get_resampled_top_edge()
        if len(top_edge) > 2:
            raise ValueError(
                'multi-segment rupture calculation has not yet been available')

        hypo = self.hypocenter
        rrup = self.surface.mesh.geodetic_min_distance(target, indices=False)
        idxs = self.surface.mesh.geodetic_min_distance(target, indices=True)
        s_lon = target.lons
        s_lat = target.lats

        # The closest points to the rupture from the sties
        cls_lon = self.surface.mesh.lons.take(idxs)
        cls_lat = self.surface.mesh.lats.take(idxs)
        cls_dep = self.surface.mesh.depths.take(idxs)

        rhypo = self.hypocenter.distance_to_mesh(target)
        rx = self.surface.get_rx_distance(target)
        rup_azimuth = numpy.empty(len(cls_lon))
        rup_distance = numpy.empty(len(cls_lon))
        for iloc, (lon, lat, dep, slon, slat) in enumerate(zip(cls_lon,
                                                               cls_lat,
                                                               cls_dep,
                                                               s_lon, s_lat)):
            # The calculation varies for different site to rupture geometries
            # The priciple is to get the rupture distance by applying the sine
            # law and the cosine rule when Rrup, dip angle, and Rhypo are
            # known.
            if rx[iloc] == 0:
                strike = self.surface.get_strike()
                azimuth = (strike + 90.0) % 360
                hdist = self.hypocenter.depth / numpy.tan(numpy.deg2rad(
                    self.surface.get_dip()))
                trace_top = hypo.point_at(
                    hdist, -self.hypocenter.depth, 360 - azimuth)
                rup_distance[iloc] = distance(
                    self.hypocenter.longitude, self.hypocenter.latitude,
                    self.hypocenter.depth, trace_top.longitude,
                    trace_top.latitude, trace_top.depth)

                rup_azimuth[iloc] = numpy.arcsin(
                    (rhypo[iloc] ** 2 - rup_distance[iloc] ** 2)
                    ** 0.5 / rhypo[iloc])

            if rx[iloc] > 0:
                rup_azimuth[iloc] = numpy.arcsin(rrup[iloc] / rhypo[iloc])
                rup_distance[iloc] = (
                    rhypo[iloc] ** 2 - rrup[iloc] ** 2) ** 0.5

            if rx[iloc] < 0:
                strike = self.surface.get_strike()
                azimuth = (strike + 90.0) % 360

                cls_point = Point(lon, lat, dep)
                hdist = dep / numpy.tan(numpy.deg2rad(self.surface.get_dip()))
                trace_top = cls_point.point_at(hdist, -dep, 360 - azimuth)
                pc_xy = get_xyz_from_ll(cls_point, hypo)
                site_xy = get_xyz_from_ll(Point(slon, slat), hypo)
                trace_top_xy = get_xyz_from_ll(trace_top, hypo)
                site_cls = (numpy.array(site_xy) - numpy.array(pc_xy))
                top_site = (numpy.array(site_xy) - numpy.array(trace_top_xy))
                phi = numpy.rad2deg(vectors2angle(site_cls, top_site))
                rup_azimuth[iloc] = numpy.arcsin(
                    rrup[iloc] / rhypo[iloc] * numpy.sin(
                        numpy.pi - numpy.deg2rad(self.surface.get_dip())
                        + numpy.deg2rad(phi)))
                rup_distance[iloc] = numpy.sin(
                    numpy.deg2rad(
                        self.surface.get_dip()) - rup_azimuth[iloc] + angle) \
                    / numpy.sin(rup_azimuth[iloc]) * rrup[iloc]
        if angle:
            return numpy.rad2deg(rup_azimuth)
        else:
            return rup_distance
