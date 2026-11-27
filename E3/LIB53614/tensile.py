#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
Synthetic tensile-test campaigns.

Each specimen comes with its measured curve (engineering strain, engineering
stress), its gauge length and cross-section area, and the fracture mode
observed on the fracture surface ("ductile": necking before fracture,
"brittle": no necking). The parameters actually used to generate the curve
(the "ground truth") are hidden in the specimen, and can be retrieved with
`truth(specimen)`, which allows one to check an identification method.

Units: stresses and moduli in MPa, lengths in mm, areas in mm², forces in N.

Material models (noiseless, in terms of the "true" strain eps, i.e., counted
from the unloaded state):
  - "elastoplastic": sigma = E eps until sigma_y, then Voce hardening
        sigma = sigma_y + Q (1 - exp(-b eps_p)), with eps_p = eps - sigma/E,
    then, beyond eps_u (onset of necking), a parabolic drop of the engineering
    stress, by a fraction alpha at fracture (eps_f). If eps_u >= eps_f, there
    is no necking.
  - "nonlinear-elastic": sigma = sigma_0 tanh(E eps / sigma_0) until fracture
    (eps_f); E is the initial tangent modulus.

Measurement model:
  - the strain is zeroed under a small preload (1–3% of the yield stress), so
    the measured curve does not pass through the origin;
  - the test is run slower in the elastic range (more points there);
  - noise on the stress (load cell) and on the strain (extensometer);
  - without extensometer, the strain is computed from the crosshead
    displacement, which includes the deformation of the machine (compliance
    C_m, in mm/N, in series with the specimen).
"""

import dataclasses
import numpy

################################################################################

MATERIALS = {
    "S355": dict(
        label="S355 structural steel",
        kind="elastoplastic",
        E=210e3, sigma_y=355., Q=170., b=20.,
        eps_u=0.16, eps_f=0.24, alpha=0.20,
        scatter=dict(E=0.02, sigma_y=0.04, Q=0.05, eps_f=0.08)),
    "Al6061": dict(
        label="6061-T6 aluminium alloy",
        kind="elastoplastic",
        E=69e3, sigma_y=275., Q=45., b=40.,
        eps_u=0.08, eps_f=0.12, alpha=0.15,
        scatter=dict(E=0.02, sigma_y=0.04, Q=0.05, eps_f=0.08)),
    "Cu": dict(
        label="Annealed copper",
        kind="elastoplastic",
        E=117e3, sigma_y=70., Q=170., b=8.,
        eps_u=0.35, eps_f=0.48, alpha=0.25,
        scatter=dict(E=0.02, sigma_y=0.06, Q=0.05, eps_f=0.08)),
    "A356": dict(
        label="A356-T6 cast aluminium alloy",
        kind="elastoplastic",
        E=72e3, sigma_y=205., Q=70., b=60.,
        eps_u=None, eps_f=0.05, alpha=0.,
        scatter=dict(E=0.02, sigma_y=0.04, Q=0.05, eps_f=0.25)),
    "CastIron": dict(
        label="Grey cast iron",
        kind="nonlinear-elastic",
        E=110e3, sigma_0=350., eps_f=0.003,
        scatter=dict(E=0.03, sigma_0=0.05, eps_f=0.15)),
    "PMMA": dict(
        label="PMMA",
        kind="nonlinear-elastic",
        E=3.2e3, sigma_0=85., eps_f=0.035,
        scatter=dict(E=0.03, sigma_0=0.04, eps_f=0.15)),
}

################################################################################

@dataclasses.dataclass
class Specimen:
    name: str               # e.g., "S355-01"
    material: str           # key of MATERIALS
    strain: numpy.ndarray   # measured engineering strain (-)
    stress: numpy.ndarray   # measured engineering stress (MPa)
    gauge_length: float     # mm
    area: float             # mm²
    extensometer: bool      # True if the strain was measured with an extensometer
    fracture: str           # observed fracture mode: "ductile" or "brittle"
    _truth: dict = dataclasses.field(repr=False, default_factory=dict)

def truth(specimen):
    """Returns the parameters actually used to generate the specimen's curve:
    E (initial tangent modulus, MPa), Rp02 (0.2% offset yield stress, MPa, None
    if no yield), Rm (tensile strength, MPa), A (strain at fracture, -),
    preload (MPa), and the model parameters."""
    return dict(specimen._truth)

################################################################################

def _stress_elastoplastic(eps, E, sigma_y, Q, b, eps_u, eps_f, alpha):
    """Engineering stress for the true strain eps (array), without necking first."""
    eps = numpy.asarray(eps, dtype=float)
    sigma = E * eps
    pl = sigma > sigma_y
    # Newton on f(s) = s - sigma_y - Q (1 - exp(-b (eps - s/E)))
    s = numpy.full(pl.sum(), sigma_y)
    e = eps[pl]
    for _ in range(50):
        ep = e - s/E
        f = s - sigma_y - Q * (1. - numpy.exp(-b*ep))
        df = 1. + Q * b * numpy.exp(-b*ep) / E
        ds = f / df
        s -= ds
        if (numpy.max(numpy.abs(ds), initial=0.) < 1e-10): break
    sigma[pl] = s
    # Necking
    if (eps_u is not None) and (eps_u < eps_f):
        nk = eps > eps_u
        s_u = _stress_elastoplastic(numpy.array([eps_u]), E, sigma_y, Q, b, None, eps_f, 0.)[0]
        sigma[nk] = s_u * (1. - alpha * ((eps[nk]-eps_u)/(eps_f-eps_u))**2)
    return sigma

def _stress_nonlinear_elastic(eps, E, sigma_0):
    return sigma_0 * numpy.tanh(E * numpy.asarray(eps, dtype=float) / sigma_0)

def _model(params):
    if (params["kind"] == "elastoplastic"):
        return lambda eps: _stress_elastoplastic(eps, params["E"], params["sigma_y"], params["Q"], params["b"], params["eps_u"], params["eps_f"], params["alpha"])
    elif (params["kind"] == "nonlinear-elastic"):
        return lambda eps: _stress_nonlinear_elastic(eps, params["E"], params["sigma_0"])
    else:
        raise ValueError(f"unknown kind: {params['kind']}")

def _Rp02(params, model):
    """0.2% offset yield stress of the noiseless curve (None if no yield)."""
    if (params["kind"] != "elastoplastic"): return None
    eps = numpy.linspace(0., params["eps_f"], 200001)
    sig = model(eps)
    g = sig - params["E"] * (eps - 0.002)
    k = numpy.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
    if (len(k) == 0): return None
    k = k[0]
    t = g[k] / (g[k] - g[k+1])
    return float(sig[k] + t * (sig[k+1] - sig[k]))

################################################################################

def make_specimen(material, name, rng, extensometer=True, gauge_length=50., area=20., machine_compliance=5e-6, n_points=1000, noise=True):
    """Generates one specimen of the given material (key of MATERIALS)."""
    nom = MATERIALS[material]
    sc = nom["scatter"]
    p = {k: v for k, v in nom.items() if k not in ("label", "scatter")}
    for k, s in sc.items():
        p[k] = nom[k] * numpy.exp(s * rng.standard_normal())
    if (p["kind"] == "elastoplastic"):
        if (nom["eps_u"] is not None):
            p["eps_u"] = nom["eps_u"] * p["eps_f"] / nom["eps_f"]
    model = _model(p)
    fracture = "ductile" if (p["kind"] == "elastoplastic" and p["eps_u"] is not None and p["eps_u"] < p["eps_f"]) else "brittle"

    # Preload: the strain is zeroed at sigma = preload (1–3% of the yield stress or of sigma_0)
    preload = rng.uniform(0.01, 0.03) * (nom["sigma_y"] if (p["kind"] == "elastoplastic") else nom["sigma_0"])
    if (p["kind"] == "elastoplastic"):
        eps_pre = preload / p["E"]
    else:
        eps_pre = p["sigma_0"] / p["E"] * numpy.arctanh(preload / p["sigma_0"])

    # Sampling: slower in the elastic range
    if (p["kind"] == "elastoplastic"):
        eps_el = min(1.5 * p["sigma_y"] / p["E"], 0.5 * p["eps_f"])
        n_el = n_points // 5
        eps = numpy.concatenate([
            numpy.linspace(eps_pre, eps_el, n_el, endpoint=False),
            numpy.linspace(eps_el, p["eps_f"], n_points - n_el)])
    else:
        eps = numpy.linspace(eps_pre, p["eps_f"], n_points)
    sig = model(eps)

    # Measured strain
    strain = eps - eps_pre
    if not (extensometer):
        strain = strain + (sig - preload) * area * machine_compliance / gauge_length
    stress = sig.copy()

    # Noise
    if (noise):
        stress += rng.normal(0., 1., size=stress.shape) * (0.3 + 0.002 * numpy.abs(stress))
        strain += rng.normal(0., 1., size=strain.shape) * (5e-6 if extensometer else 2e-5)

    t = dict(
        E=float(p["E"]),
        Rp02=_Rp02(p, model),
        Rm=float(numpy.max(sig)),
        A=float(eps[-1] - eps_pre),
        preload=float(preload),
        params={k: (float(v) if isinstance(v, (float, numpy.floating)) else v) for k, v in p.items()})
    return Specimen(
        name=name,
        material=material,
        strain=strain,
        stress=stress,
        gauge_length=float(gauge_length),
        area=float(area),
        extensometer=bool(extensometer),
        fracture=fracture,
        _truth=t)

def campaign(n_per_material=3, materials=None, seed=0, extensometer=True, gauge_lengths=(50.,), area=20., machine_compliance=5e-6, n_points=1000, noise=True):
    """Generates a tensile-test campaign: n_per_material specimens of each
    material (all materials of MATERIALS by default) for each gauge length.
    Returns a list of Specimen."""
    rng = numpy.random.default_rng(seed)
    if (materials is None): materials = list(MATERIALS.keys())
    specimens = []
    for material in materials:
        k = 0
        for gauge_length in gauge_lengths:
            for _ in range(n_per_material):
                k += 1
                specimens.append(make_specimen(
                    material=material,
                    name=f"{material}-{k:02d}",
                    rng=rng,
                    extensometer=extensometer,
                    gauge_length=gauge_length,
                    area=area,
                    machine_compliance=machine_compliance,
                    n_points=n_points,
                    noise=noise))
    return specimens
