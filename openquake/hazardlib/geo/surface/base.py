# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2016 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`openquake.hazardlib.geo.surface.base` implements
:class:`BaseSurface` and :class:`BaseQuadrilateralSurface`.
"""
import abc

import numpy
import math
from openquake.hazardlib.geo import geodetic, utils, Point, Line
from openquake.baselib.python3compat import with_metaclass


class BaseSurface(with_metaclass(abc.ABCMeta)):
    """
    Base class for a surface in 3D-space.
    """

    @abc.abstractmethod
    def get_min_distance(self, mesh):
        """
        Compute and return the minimum distance from the surface to each point
        of ``mesh``. This distance is sometimes called ``Rrup``.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            minimum distance to.
        :returns:
            A numpy array of distances in km.
        """

    @abc.abstractmethod
    def get_closest_points(self, mesh):
        """
        For each point from ``mesh`` find a closest point belonging to surface.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to find
            closest points to.
        :returns:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of the same shape as
            ``mesh`` with closest surface's points on respective indices.
        """

    @abc.abstractmethod
    def get_joyner_boore_distance(self, mesh):
        """
        Compute and return Joyner-Boore (also known as ``Rjb``) distance
        to each point of ``mesh``.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            Joyner-Boore distance to.
        :returns:
            Numpy array of closest distances between the projections of surface
            and each point of the ``mesh`` to the earth surface.
        """

    @abc.abstractmethod
    def get_ry0_distance(self, mesh):
        """
        Compute the minimum distance between each point of a mesh and the great
        circle arcs perpendicular to the average strike direction of the
        fault trace and passing through the end-points of the trace.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            Ry0-distance to.
        :returns:
            Numpy array of distances in km.
        """

    @abc.abstractmethod
    def get_rx_distance(self, mesh):
        """
        Compute distance between each point of mesh and surface's great circle
        arc.

        Distance is measured perpendicular to the rupture strike, from
        the surface projection of the updip edge of the rupture, with
        the down dip direction being positive (this distance is usually
        called ``Rx``).

        In other words, is the horizontal distance to top edge of rupture
        measured perpendicular to the strike. Values on the hanging wall
        are positive, values on the footwall are negative.

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            Rx-distance to.
        :returns:
            Numpy array of distances in km.
        """

    @abc.abstractmethod
    def get_top_edge_depth(self):
        """
        Compute minimum depth of surface's top edge.

        :returns:
            Float value, the vertical distance between the earth surface
            and the shallowest point in surface's top edge in km.
        """

    @abc.abstractmethod
    def get_strike(self):
        """
        Compute surface's strike as decimal degrees in a range ``[0, 360)``.

        The actual definition of the strike might depend on surface geometry.

        :returns:
            Float value, the azimuth (in degrees) of the surface top edge
        """

    @abc.abstractmethod
    def get_dip(self):
        """
        Compute surface's dip as decimal degrees in a range ``(0, 90]``.

        The actual definition of the dip might depend on surface geometry.

        :returns:
            Float value, the inclination (in degrees) of the surface with
            respect to the Earth surface
        """

    @abc.abstractmethod
    def get_width(self):
        """
        Compute surface's width (that is surface extension along the
        dip direction) in km.

        The actual definition depends on the type of surface geometry.

        :returns:
            Float value, the surface width
        """

    @abc.abstractmethod
    def get_area(self):
        """
        Compute surface's area in squared km.

        :returns:
            Float value, the surface area
        """

    @abc.abstractmethod
    def get_bounding_box(self):
        """
        Compute surface geographical bounding box.

        :return:
            A tuple of four items. These items represent western, eastern,
            northern and southern borders of the bounding box respectively.
            Values are floats in decimal degrees.
        """

    @abc.abstractmethod
    def get_middle_point(self):
        """
        Compute coordinates of surface middle point.

        The actual definition of ``middle point`` depends on the type of
        surface geometry.

        :return:
            instance of :class:`openquake.hazardlib.geo.point.Point`
            representing surface middle point.
        """


class BaseQuadrilateralSurface(with_metaclass(abc.ABCMeta, BaseSurface)):
    """
    Base class for a quadrilateral surface in 3D-space.

    Subclasses must implement :meth:`_create_mesh`, and superclass methods
    :meth:`get_strike() <.base.BaseSurface.get_strike>`,
    :meth:`get_dip() <.base.BaseSurface.get_dip>` and
    :meth:`get_width() <.base.BaseSurface.get_width>`,
    and can override any others just for the sake of performance
    """

    def __init__(self):
        self._mesh = None

    def get_min_distance(self, mesh):
        """
        See :meth:`superclass method
        <.base.BaseSurface.get_min_distance>`
        for spec of input and result values.

        Base class implementation calls the :meth:`corresponding
        <openquake.hazardlib.geo.mesh.Mesh.get_min_distance>` method of the
        surface's :meth:`mesh <get_mesh>`.

        Subclasses may override this method in order to make use
        of knowledge of a specific surface shape and thus perform
        better.
        """
        return self.get_mesh().get_min_distance(mesh)

    def get_closest_points(self, mesh):
        """
        See :meth:`superclass method
        <.base.BaseSurface.get_closest_points>`
        for spec of input and result values.

        Base class implementation calls the :meth:`corresponding
        <openquake.hazardlib.geo.mesh.Mesh.get_closest_points>` method of the
        surface's :meth:`mesh <get_mesh>`.
        """
        return self.get_mesh().get_closest_points(mesh)

    def get_joyner_boore_distance(self, mesh):
        """
        See :meth:`superclass method
        <.base.BaseSurface.get_joyner_boore_distance>`
        for spec of input and result values.

        Base class calls surface mesh's method
        :meth:`~openquake.hazardlib.geo.mesh.Mesh.get_joyner_boore_distance`.
        """
        return self.get_mesh().get_joyner_boore_distance(mesh)

    def get_ry0_distance(self, mesh):
        """
        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points to calculate
            Ry0-distance to.
        :returns:
            Numpy array of distances in km.

        See also :meth:`superclass method <.base.BaseSurface.get_ry0_distance>`
        for spec of input and result values.

        This method uses an average strike direction to compute ry0.
        """
        # This computes ry0 by using an average strike direction
        top_edge = self.get_mesh()[0:1]
        mean_strike = self.get_strike()

        dst1 = geodetic.distance_to_arc(top_edge.lons[0, 0],
                                        top_edge.lats[0, 0],
                                        (mean_strike + 90.) % 360,
                                        mesh.lons, mesh.lats)

        dst2 = geodetic.distance_to_arc(top_edge.lons[0, -1],
                                        top_edge.lats[0, -1],
                                        (mean_strike + 90.) % 360,
                                        mesh.lons, mesh.lats)
        # Find the points on the rupture

        # Get the shortest distance from the two lines
        idx = numpy.sign(dst1) == numpy.sign(dst2)
        dst = numpy.zeros_like(dst1)
        dst[idx] = numpy.fmin(numpy.abs(dst1[idx]), numpy.abs(dst2[idx]))

        return dst

    def get_rx_distance(self, mesh):
        """
        See :meth:`superclass method
        <.base.BaseSurface.get_rx_distance>`
        for spec of input and result values.

        The method extracts the top edge of the surface. For each point in mesh
        it computes the Rx distance to each segment the top edge is made
        of. The calculation is done by calling the function
        :func:`openquake.hazardlib.geo.geodetic.distance_to_arc`. The final Rx
        distance matrix is then constructed by taking, for each point in mesh,
        the minimum Rx distance value computed.
        """
        top_edge = self.get_mesh()[0:1]

        dists = []
        if top_edge.lons.shape[1] < 3:

            i = 0
            p1 = Point(
                top_edge.lons[0, i],
                top_edge.lats[0, i],
                top_edge.depths[0, i]
            )
            p2 = Point(
                top_edge.lons[0, i + 1], top_edge.lats[0, i + 1],
                top_edge.depths[0, i + 1]
            )
            azimuth = p1.azimuth(p2)
            dists.append(
                geodetic.distance_to_arc(
                    p1.longitude, p1.latitude, azimuth,
                    mesh.lons, mesh.lats
                )
            )

        else:

            for i in range(top_edge.lons.shape[1] - 1):
                p1 = Point(
                    top_edge.lons[0, i],
                    top_edge.lats[0, i],
                    top_edge.depths[0, i]
                )
                p2 = Point(
                    top_edge.lons[0, i + 1],
                    top_edge.lats[0, i + 1],
                    top_edge.depths[0, i + 1]
                )
                # Swapping
                if i == 0:
                    pt = p1
                    p1 = p2
                    p2 = pt

                # Computing azimuth and distance
                if i == 0 or i == top_edge.lons.shape[1] - 2:
                    azimuth = p1.azimuth(p2)
                    tmp = geodetic.distance_to_semi_arc(p1.longitude,
                                                        p1.latitude,
                                                        azimuth,
                                                        mesh.lons, mesh.lats)
                else:
                    tmp = geodetic.min_distance_to_segment([p1.longitude,
                                                            p2.longitude],
                                                           [p1.latitude,
                                                            p2.latitude],
                                                           mesh.lons,
                                                           mesh.lats)
                # Correcting the sign of the distance
                if i == 0:
                    tmp *= -1
                dists.append(tmp)

        # Computing distances
        dists = numpy.array(dists)
        iii = abs(dists).argmin(axis=0)
        dst = dists[iii, list(range(dists.shape[1]))]

        return dst

    def get_top_edge_depth(self):
        """
        Return minimum depth of surface's top edge.

        :returns:
            Float value, the vertical distance between the earth surface
            and the shallowest point in surface's top edge in km.
        """
        top_edge = self.get_mesh()[0:1]
        if top_edge.depths is None:
            return 0
        else:
            return numpy.min(top_edge.depths)

    def _get_top_edge_centroid(self):
        """
        Return :class:`~openquake.hazardlib.geo.point.Point` representing the
        surface's top edge centroid.
        """
        top_edge = self.get_mesh()[0:1]
        return top_edge.get_middle_point()

    def get_mesh(self):
        """
        Return surface's mesh.

        Uses :meth:`_create_mesh` for creating the mesh for the first time.
        All subsequent calls to :meth:`get_mesh` return the same mesh object.

        .. warning::
            It is required that the mesh is constructed "top-to-bottom".
            That is, the first row of points should be the shallowest.
        """
        if self._mesh is None:
            self._mesh = self._create_mesh()
            assert (
                self._mesh.depths is None or len(self._mesh.depths) == 1
                or self._mesh.depths[0][0] < self._mesh.depths[-1][0]
            ), "the first row of points in the mesh must be the shallowest"
        return self._mesh

    def get_area(self):
        """
        Compute area as the sum of the mesh cells area values.
        """
        mesh = self.get_mesh()
        _, _, _, area = mesh.get_cell_dimensions()

        return numpy.sum(area)

    def get_bounding_box(self):
        """
        Compute surface bounding box from surface mesh representation. That is
        extract longitudes and latitudes of mesh points and calls:
        :meth:`openquake.hazardlib.geo.utils.get_spherical_bounding_box`

        :return:
            A tuple of four items. These items represent western, eastern,
            northern and southern borders of the bounding box respectively.
            Values are floats in decimal degrees.
        """
        mesh = self.get_mesh()

        return utils.get_spherical_bounding_box(mesh.lons, mesh.lats)

    def get_middle_point(self):
        """
        Compute middle point from surface mesh representation. Calls
        :meth:`openquake.hazardlib.geo.mesh.RectangularMesh.get_middle_point`
        """
        mesh = self.get_mesh()

        return mesh.get_middle_point()

    def get_resampled_top_edge(self, return_top_edge_index=False, angle_var=0.1):
        """
        This methods computes a simplified representation of a fault top edge
        by removing the points that are not describing a change of direction,
        provided a certain tolerance angle.

        :param return_top_edge_index:
            If ``True`` , the indices of the top edge on the rupture mesh are
            return.
        :param float angle_var:
            Number representing the maximum deviation (in degrees) admitted
            without the creation of a new segment
        :returns:
            line_top_edge, an instance
            :class:`~openquake.hazardlib.geo.line.Line`
            representing the rupture surface's top edge.
            top_edge_index, a numpy array of floats represents the indices of
            the top edge on the rupture mesh(only be return when
            return_top_edge_index is ``True``).
        """
        mesh = self.get_mesh()
        top_edge = [Point(mesh.lons[0][0], mesh.lats[0][0], mesh.depths[0][0])]
        top_edge_index = [0]

        for i in range(len(mesh.triangulate()[1][0]) - 1):
            v1 = numpy.asarray(mesh.triangulate()[1][0][i])
            v2 = numpy.asarray(mesh.triangulate()[1][0][i + 1])
            cosang = numpy.dot(v1, v2)
            sinang = numpy.linalg.norm(numpy.cross(v1, v2))
            angle = math.degrees(numpy.arctan2(sinang, cosang))

            if abs(angle) > angle_var:

                top_edge.append(Point(mesh.lons[0][i + 1],
                                      mesh.lats[0][i + 1],
                                      mesh.depths[0][i + 1]))

                top_edge_index.append(i)
                top_edge.append(Point(mesh.lons[0][-1],
                                      mesh.lats[0][-1], mesh.depths[0][-1]))
                line_top_edge = Line(top_edge)

        top_edge.append(Point(mesh.lons[0][-1],
                              mesh.lats[0][-1], mesh.depths[0][-1]))
        line_top_edge = Line(top_edge)
        if return_top_edge_index:
            top_edge_index.append(len(mesh.lons[0]) - 1)
            return line_top_edge, top_edge_index

        return line_top_edge

    @abc.abstractmethod
    def _create_mesh(self):
        """
        Create and return the mesh of points covering the surface.

        :returns:
            An instance of
            :class:`openquake.hazardlib.geo.mesh.RectangularMesh`.
        """

    def get_hypo_location(self, mesh_spacing, hypo_loc=None):
        """
        The method determines the location of the hypocentre within the rupture

        :param mesh:
            :class:`~openquake.hazardlib.geo.mesh.Mesh` of points
        :param mesh_spacing:
            The desired distance between two adjacent points in source's
            ruptures' mesh, in km. Mainly this parameter allows to balance
            the trade-off between time needed to compute the distance
            between the rupture surface and a site and the precision of that
            computation.
        :param hypo_loc:
            Hypocentre location as fraction of rupture plane, as a tuple of
            (Along Strike, Down Dip), e.g. a hypocentre located in the centroid
            of the rupture would be input as (0.5, 0.5), whereas a
            hypocentre located in a position 3/4 along the length, and 1/4 of
            the way down dip of the rupture plane would be entered as
            (0.75, 0.25).
        :returns:
            Hypocentre location as instance of
            :class:`~openquake.hazardlib.geo.point.Point`
        """
        mesh = self.get_mesh()
        centroid = mesh.get_middle_point()
        if hypo_loc is None:
            return centroid

        total_len_y = (len(mesh.depths) - 1) * mesh_spacing
        y_distance = hypo_loc[1] * total_len_y
        y_node = int(numpy.round(y_distance / mesh_spacing))
        total_len_x = (len(mesh.lons[y_node]) - 1) * mesh_spacing
        x_distance = hypo_loc[0] * total_len_x
        x_node = int(numpy.round(x_distance / mesh_spacing))
        hypocentre = Point(mesh.lons[y_node][x_node],
                           mesh.lats[y_node][x_node],
                           mesh.depths[y_node][x_node])
        return hypocentre
