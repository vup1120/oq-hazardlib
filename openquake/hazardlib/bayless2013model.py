# Import methods
import numpy as np

import shakemap.grind.fault as fault
import shakemap.grind.ecef as ecef
from shakemap.grind.vector import Vector
from shakemap.grind.distance import get_distance
from shakemap.grind.distance import distance_sq_to_segment


from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib import const
from openquake.hazardlib.geo import RectangularMesh
from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo.surface.planar import PlanarSurface


import pyprind


def setPseudoHypo(i, surface, hypo):
    """
    Adapted from ShakeMap 3.5 src/contour/directivity.c 
    From Bayless and Somerville:
        "Define the pseudo-hypocenter for rupture of successive segments as 
         the point on the side edge of the fault segment that is closest to 
         the side edge of the previous segment, and that lies half way 
         between the top and bottom of the fault. We assume that the fault is
         segmented along strike, not updip. All geometric parameters are 
         computed relative to the pseudo-hypocenter."
     Simple version by Yen-Shin
    """
    index_patch = surface.hypocentre_patch_index(
        hypo, surface.get_resampled_top_edge(),
        surface.mesh.depths[0][0], surface.mesh.depths[-1][0],
        surface.get_dip())
    if i == index_patch:
        phyp = hypo
    elif i < index_patch:
        row = np.round(len(surface.mesh.depths[:,0])/2)
        tmp, col = surface.get_resampled_top_edge(return_top_edge_index=True)
        phyp = Point(surface.mesh.lons[row][col[i+1]],
                     surface.mesh.lats[row][col[i+1]],
                     surface.mesh.depths[row][col[i+1]])
    elif i > index_patch:
        row = np.round(len(surface.mesh.depths[:,0])/2)
        tmp, col = surface.get_resampled_top_edge(return_top_edge_index=True)
        phyp = Point(surface.mesh.lons[row][col[i]],
                     surface.mesh.lats[row][col[i]],
                     surface.mesh.depths[row][col[i]])
    return phyp

def computeThetaAndS(phyp, P0, P1, P2, P3, sites):
    """
    :param i:
        Compute d for the i-th quad/segment. 
    """
    # self.phyp is in ECEF

    epi_ecef = Vector.fromPoint(Point(phyp.longitude, phyp.latitude, 0.0))
    epi_col = np.array([[epi_ecef.x], [epi_ecef.y], [epi_ecef.z]])

    # First compute along strike vector
    p0 = Vector.fromPoint(P0) # convert to ECEF
    p1 = Vector.fromPoint(P1)
    e01 = p1 - p0
    e01norm = e01.norm()
    hp0 = p0 - epi_ecef
    hp1 = p1 - epi_ecef
    strike_min = Vector.dot(hp0, e01norm)/1000.0 # convert to km
    strike_max = Vector.dot(hp1, e01norm)/1000.0 # convert to km 
    strike_col = np.array([[e01norm.x],[e01norm.y],[e01norm.z]]) # ECEF coords

    # Sites
    slat = sites.lats
    slon = sites.lons

    # Convert sites to ECEF:
    site_ecef_x = np.ones_like(slat)
    site_ecef_y = np.ones_like(slat)
    site_ecef_z = np.ones_like(slat)

    # Make a 3x(#number of sites) matrix of site locations
    # (rows are x, y, z) in ECEF
    site_ecef_x, site_ecef_y, site_ecef_z = ecef.latlon2ecef(
        slat, slon, np.zeros(slon.shape) )
    site_mat = np.array([np.reshape(site_ecef_x, (-1,)),
                         np.reshape(site_ecef_y, (-1,)),
                         np.reshape(site_ecef_z, (-1,))])

    # Epicenter-to-site matrix
    e2s_mat = site_mat - epi_col # in ECEF
    mag = np.sqrt(np.sum(e2s_mat*e2s_mat, axis = 0))

    # Avoid division by zero
    mag[mag == 0] = 1e-12
    e2s_norm = e2s_mat/mag

    # Dot epicenter-to-site with along-strike vector
    s_raw = np.sum(e2s_mat * strike_col, axis = 0)/1000.0 # conver to km

    # Put back into a 2d array
    s_raw = np.reshape(s_raw, len(sites))
    s = np.abs(s_raw.clip(min = strike_min,
                               max = strike_max)).clip(min = np.exp(1))
    # Compute theta
    sdots = np.sum(e2s_norm * strike_col, axis = 0)
    theta_raw = np.arccos(sdots)

    # But theta is defined to be the reference angle
    # (i.e., the equivalent angle between 0 and 90 deg)
    sintheta = np.abs(np.sin(theta_raw))
    costheta = np.abs(np.cos(theta_raw))
    theta = np.arctan2(sintheta, costheta)
    theta = np.reshape(theta, len(sites))
    return s, theta

def computeSS(s, theta, sites, L, T_Mw, Rrup):
    # s is the length of striking fault rupturing toward site; max[(X*L),exp(1)]
    # theta (see Figure 5 in SSGA97)

    # Geometric directivity predictor:
    f_geom = np.log(s) * (0.5 * np.cos(2*theta) + 0.5)

    # Distance taper
    T_CD = np.ones_like(sites.lons)
    ix = [(Rrup/L > 0.5) & (Rrup/L < 1.0)]
    T_CD[ix] = 1 - (Rrup[ix]/L - 0.5)/0.5
    T_CD[Rrup/L >= 1.0 ] = 0.0

    # Azimuth taper
    T_Az = 1.0

    tapering = T_CD * T_Mw * T_Az
    return f_geom, tapering

def computeD(phyp, P0,P1,P2,P3, sites):
    """
    :param i:
        Compute d for the i-th quad/segment. 
    Y = d/W, where d is the portion (in km) of the width of the fault which 
    ruptures up-dip from the hypocenter to the top of the fault.
    """
    hyp_ecef = Vector.fromPoint(phyp) # already in ECEF
    hyp_col = np.array([[hyp_ecef.x], [hyp_ecef.y], [hyp_ecef.z]])

    # First compute "updip" vector
    p1 = Vector.fromPoint(P1) # convert to ECEF
    p2 = Vector.fromPoint(P2)
    e21 = p1 - p2
    e21norm = e21.norm()
    hp1 = p1 - hyp_ecef
    udip_len = Vector.dot(hp1, e21norm)/1000.0 # convert to km (used as max later)
    udip_col = np.array([[e21norm.x], [e21norm.y], [e21norm.z]]) # ECEF coords

    # Sites
    slat = sites.lats
    slon = sites.lons

    # Convert sites to ECEF:
    site_ecef_x = np.ones_like(slat)
    site_ecef_y = np.ones_like(slat)
    site_ecef_z = np.ones_like(slat)

    # Make a 3x(#number of sites) matrix of site locations
    # (rows are x, y, z) in ECEF
    site_ecef_x, site_ecef_y, site_ecef_z = ecef.latlon2ecef(
        slat, slon, np.zeros(slon.shape) )
    site_mat = np.array([np.reshape(site_ecef_x, (-1,)),
                         np.reshape(site_ecef_y, (-1,)),
                         np.reshape(site_ecef_z, (-1,))])

    # Hypocenter-to-site matrix
    h2s_mat = site_mat - hyp_col # in ECEF

    # Dot hypocenter-to-site with updip vector
    d_raw = np.abs(np.sum(h2s_mat * udip_col, axis = 0))/1000.0 # convert to km
    d_raw = np.reshape(d_raw, len(sites))
    d = d_raw.clip(min = 1.0, max = udip_len)
    return d

def computeDS(d, Az, T_Mw, Rx, Rrup, W, sites):
    # d is the length of dipping fault rupturing toward site;
    # Note: max[(Y*W),exp(0)] -- just apply a min of 1?

    # Geometric directivity predictor:
    RxoverW = (Rx / W).clip(min = -np.pi/2.0, max = 2.0*np.pi/3.0)
    RxoverW = RxoverW.reshape(d.shape)
    f_geom = np.log(d) * np.cos(RxoverW)
    # Distance taper
    T_CD = np.ones_like(len(sites))
    ix = [(Rrup/W > 1.5) & (Rrup/W < 2.0)]
    T_CD = 1.0 - (Rrup/W - 1.5)/0.5
    T_CD[Rrup/W >= 2.0 ] = 0.0

    # Azimuth taper
    T_Az = np.sin(np.abs(Az))**2
    tapering = T_CD * T_Mw * T_Az
    return f_geom, tapering

def computeAz(Rx, Ry):
    Az = np.ones_like(Rx) * np.pi/2.0
    Az = Az * np.sign(Rx)
    ix = [Ry > 0.0]
    Az[ix] = np.arctan(Rx[ix]/Ry[ix])
    Az = Az
    return Az

def Magnitude_taper(M):
    if M <= 5.0: 
        T_Mw = 0.0
    elif (M > 5.0) and (M < 6.5):
        T_Mw = 1.0 - (6.5 - M)/1.5
    else:
        T_Mw = 1.0
    return T_Mw

def getSlipCategory(rake):
    """
    Sets self.SlipCategory based on rake angle. Can be SS for 
    strike slip, DS for dip slip, or Unspecified. 
    """
    arake = np.abs(rake)
    SlipCategory = 'Unspecified'
    if ((arake >=  0) and (arake <= 30)) or ((arake >= 150) and (arake <= 180)):
        SlipCategory = 'SS'
    if (arake >= 60) and (arake <= 120):
        SlipCategory = 'DS'
    return SlipCategory

def fd(rake, fd_SS, fd_DS):
    # Normalize rake to reference angle
    sintheta = np.abs(np.sin(np.radians(rake)))
    costheta = np.abs(np.cos(np.radians(rake)))
    refrake = np.arctan2(sintheta, costheta)

    # Compute weights:
    DipWeight = refrake/(np.pi/2.0)
    StrikeWeight = 1.0 - DipWeight
    fdcombined = StrikeWeight*fd_SS + DipWeight*self.fd_DS
    fd = fd + weights*fdcombined
    return fd

def getFinalFDforObligateFault(rake, fd_SS, fd_DS):               
    # Normalize rake to reference angle
    sintheta = np.abs(np.sin(np.radians(rake)))
    costheta = np.abs(np.cos(np.radians(rake)))
    refrake = np.arctan2(sintheta, costheta)

    # Compute weights:
    DipWeight = refrake/(np.pi/2.0)
    StrikeWeight = 1.0 - DipWeight
    fdcombined = StrikeWeight*fd_SS + DipWeight*fd_DS
    return fdcombined
