#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
Synthetic data for the tip deflection of a cantilever beam (Euler–Bernoulli),
of length L, rectangular cross-section b × h, Young modulus E, under a tip
load P:
    delta = P L³ / (3 E I), with I = b h³ / 12.

Units: SI (N, m, Pa).
"""

import numpy

INPUTS = ("P", "L", "E", "b", "h")

RANGES = dict(
    P=(1., 10.),         # N
    L=(0.1, 0.5),        # m
    E=(70e9, 210e9),     # Pa
    b=(0.01, 0.05),      # m
    h=(0.01, 0.03))      # m

def deflection(P, L, E, b, h):
    """Tip deflection (m) of the cantilever."""
    I = b * h**3 / 12
    return P * L**3 / (3 * E * I)

def samples(n, ranges=None, noise=0.02, seed=None):
    """Returns X (n×5 array, columns P, L, E, b, h, sampled uniformly in their
    ranges) and delta (n array, tip deflections with a relative Gaussian noise
    of standard deviation `noise`). `ranges` (dict) overrides some of the
    default RANGES."""
    rng = numpy.random.default_rng(seed)
    r = dict(RANGES)
    if (ranges is not None): r.update(ranges)
    X = numpy.stack([rng.uniform(*r[k], size=n) for k in INPUTS], axis=1)
    delta = deflection(*X.T) * (1. + noise * rng.standard_normal(n))
    return X, delta
