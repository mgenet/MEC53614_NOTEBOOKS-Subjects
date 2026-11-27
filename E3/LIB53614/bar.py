#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
Synthetic full-field (DIC-like) measurements on a heterogeneous bar in tension.

The bar (length `length`, cross-section area `area`) is clamped at x = 0 and
pulled by a force `force` at x = length. Its Young modulus varies along the
bar: a slight linear gradient (e.g., due to the printing direction of an
additively manufactured part) and a softer zone (e.g., a lack-of-fusion
porosity cluster):
    E(x) = E_0 (1 + g (x/length - 1/2)) (1 - d_0 exp(-(x - x_0)² / (2 w²))).
Since the stress sigma = force/area is uniform, the displacement is
    u(x) = sigma ∫_0^x 1/E(s) ds.
The displacement is "measured" at n_points equally spaced points (including
both ends), with an additive Gaussian noise of standard deviation `noise`.

The parameters actually used (the "truth") can be retrieved with
`truth(bar)`, whose entry "E" is a function of x.

Units: N, mm, MPa.
"""

import dataclasses
import numpy

################################################################################

@dataclasses.dataclass
class Bar:
    x: numpy.ndarray    # measurement points (mm)
    u: numpy.ndarray    # measured displacements (mm)
    length: float       # mm
    area: float         # mm²
    force: float        # N
    noise: float        # standard deviation of the measurement noise (mm)
    _truth: dict = dataclasses.field(repr=False, default_factory=dict)

def _E(x, E_0, g, d_0, x_0, w, length):
    x = numpy.asarray(x, dtype=float)
    return E_0 * (1. + g * (x/length - 0.5)) * (1. - d_0 * numpy.exp(-(x - x_0)**2 / (2 * w**2)))

def truth(bar):
    """Returns the parameters actually used to generate the bar, and the
    Young modulus field as a function: truth(bar)["E"](x) (MPa)."""
    p = dict(bar._truth)
    params = dict(p)
    p["E"] = lambda x: _E(x, length=bar.length, **params)
    return p

def specimen(seed=0, length=100., area=20., force=200., E_0=3500., n_points=101, noise=5e-4, d_0=None, w=None, x_0=None, g=None):
    """Generates one bar. The defect parameters (depth d_0, width w, position
    x_0) and the gradient g are random (given the seed), unless specified."""
    rng = numpy.random.default_rng(seed)
    p = dict(
        E_0=float(E_0),
        g=float(rng.uniform(-0.06, 0.06) if (g is None) else g),
        d_0=float(rng.uniform(0.3, 0.5) if (d_0 is None) else d_0),
        x_0=float(rng.uniform(0.3, 0.7) * length if (x_0 is None) else x_0),
        w=float(rng.uniform(3., 8.) if (w is None) else w))

    # Exact displacements (fine trapezoidal integration of the compliance)
    x_fine = numpy.linspace(0., length, 100*(n_points-1)+1)
    c_fine = 1. / _E(x_fine, length=length, **p)
    u_fine = numpy.concatenate([[0.], numpy.cumsum(0.5 * (c_fine[1:] + c_fine[:-1]) * numpy.diff(x_fine))]) * force / area
    x = x_fine[::100]
    u = u_fine[::100].copy()

    # Noise
    u += noise * rng.standard_normal(u.shape)

    return Bar(x=x, u=u, length=float(length), area=float(area), force=float(force), noise=float(noise), _truth=p)
