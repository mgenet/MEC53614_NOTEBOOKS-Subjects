#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
Synthetic compression tests on pin-ended steel columns (struts), labelled by
their outcome: 0 = "safe" (the column carries the load), 1 = "plastic
collapse", 2 = "elastic buckling".

Each test is described by the slenderness of the column, lmbda = L/i (L: length,
i: radius of gyration of the cross-section), and by the applied mean stress
sigma = P/A (MPa).

Ground truth. Each column has its own yield stress sigma_y (log-normal scatter
around a mean above the nominal value, as for real steel) and its own
geometric imperfection, which reduces its strength according to the
Ayrton-Perry formula:
    sigma_E = pi² E / lmbda²                    (Euler stress)
    lmbda_bar = sqrt(sigma_y / sigma_E)         (relative slenderness)
    eta = alpha (lmbda_bar - 0.2)_+             (generalized imperfection)
    phi = (1 + eta + lmbda_bar²) / 2
    sigma_f = sigma_y / (phi + sqrt(phi² - lmbda_bar²))   (failure stress)
with an imperfection factor alpha that also varies from column to column. The
column fails if sigma >= sigma_f. A failed column is labelled "elastic
buckling" if its Euler stress is below its yield stress (lmbda_bar > 1), and
"plastic collapse" otherwise (this is a convention: around lmbda_bar = 1, the
failure is in fact an interaction of both).

For a perfect column (alpha = 0) and a given sigma_y, the boundaries are
sigma = sigma_y (squash load) and sigma = sigma_E (Euler), which are straight
lines in the (log lmbda, log sigma) plane, meeting at lmbda_c = pi sqrt(E / sigma_y).

The proven design method is the Eurocode 3 buckling curve, `ec3_stress`: the same
Ayrton-Perry formula, with the nominal yield stress and a code value of alpha
(0.21, 0.34, 0.49 or 0.76 for curves a, b, c, d).

The parameters actually used (the "truth") can be retrieved with
`truth(campaign)`.

Units: MPa.
"""

import dataclasses
import numpy

################################################################################

CLASSES = ("safe", "plastic collapse", "elastic buckling")

E_NOMINAL = 210e3       # MPa
SIGMA_Y_NOMINAL = 355.  # MPa (S355)

@dataclasses.dataclass
class Campaign:
    lmbda: numpy.ndarray    # slenderness L/i (-)
    sigma: numpy.ndarray    # applied mean stress P/A (MPa)
    label: numpy.ndarray    # 0: safe, 1: plastic collapse, 2: elastic buckling
    _truth: dict = dataclasses.field(repr=False, default_factory=dict)

def euler_stress(lmbda, E=E_NOMINAL):
    """Euler (critical) stress of a perfect pin-ended column (MPa)."""
    lmbda = numpy.asarray(lmbda, dtype=float)
    return numpy.pi**2 * E / lmbda**2

def ayrton_perry_stress(lmbda, sigma_y, alpha, E=E_NOMINAL):
    """Failure stress of an imperfect column, by the Ayrton-Perry formula (MPa)."""
    lmbda = numpy.asarray(lmbda, dtype=float)
    lmbda_bar = numpy.sqrt(sigma_y / euler_stress(lmbda, E))
    eta = alpha * numpy.maximum(lmbda_bar - 0.2, 0.)
    phi = 0.5 * (1. + eta + lmbda_bar**2)
    return sigma_y / (phi + numpy.sqrt(phi**2 - lmbda_bar**2))

def ec3_stress(lmbda, alpha=0.34, sigma_y=SIGMA_Y_NOMINAL, E=E_NOMINAL):
    """Design buckling stress of the Eurocode 3 (chi f_y, without partial
    safety factor), for the buckling curve of imperfection factor alpha
    (0.21: curve a, 0.34: curve b, 0.49: curve c, 0.76: curve d) (MPa)."""
    return ayrton_perry_stress(lmbda, sigma_y, alpha, E)

def truth(campaign):
    """Returns the parameters actually used to generate the campaign: the yield
    stress "sigma_y", imperfection factor "alpha", Euler stress "sigma_E" and
    failure stress "sigma_f" of each column (arrays), and the Young modulus "E"."""
    return dict(campaign._truth)

def failure_probability(lmbda, sigma, n=200000, seed=12345, E=E_NOMINAL, sigma_y_mean=380., sigma_y_cov=0.07, alpha_median=0.2, alpha_cov=0.3):
    """True probability that a column of slenderness lmbda (scalar), taken at
    random from the production (same distributions of yield stress and
    imperfection as in `campaign`), fails under the applied stress sigma
    (scalar, MPa). For checking only: in practice, it is unknown."""
    rng = numpy.random.default_rng(seed)
    s = numpy.sqrt(numpy.log(1. + sigma_y_cov**2))
    sigma_y = sigma_y_mean * numpy.exp(s * rng.standard_normal(n) - s**2/2)
    s = numpy.sqrt(numpy.log(1. + alpha_cov**2))
    alpha = alpha_median * numpy.exp(s * rng.standard_normal(n))
    return float(numpy.mean(sigma >= ayrton_perry_stress(lmbda, sigma_y, alpha, E)))

def campaign(n=300, seed=0, lmbda_range=(15., 250.), sigma_range=(20., 800.), E=E_NOMINAL, sigma_y_mean=380., sigma_y_cov=0.07, alpha_median=0.2, alpha_cov=0.3):
    """Generates n tests: slenderness and applied stress are sampled
    log-uniformly in the given ranges; yield stress and imperfection factor are
    log-normal (mean / median and coefficient of variation given)."""
    rng = numpy.random.default_rng(seed)
    lmbda = numpy.exp(rng.uniform(*numpy.log(lmbda_range), size=n))
    sigma = numpy.exp(rng.uniform(*numpy.log(sigma_range), size=n))
    s = numpy.sqrt(numpy.log(1. + sigma_y_cov**2))
    sigma_y = sigma_y_mean * numpy.exp(s * rng.standard_normal(n) - s**2/2)
    s = numpy.sqrt(numpy.log(1. + alpha_cov**2))
    alpha = alpha_median * numpy.exp(s * rng.standard_normal(n))
    sigma_E = euler_stress(lmbda, E)
    sigma_f = ayrton_perry_stress(lmbda, sigma_y, alpha, E)
    label = numpy.where(sigma < sigma_f, 0, numpy.where(sigma_E < sigma_y, 2, 1)).astype(int)
    return Campaign(lmbda=lmbda, sigma=sigma, label=label, _truth=dict(E=float(E), sigma_y=sigma_y, alpha=alpha, sigma_E=sigma_E, sigma_f=sigma_f))
