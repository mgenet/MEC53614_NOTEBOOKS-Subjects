#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
A finite-width plate with a central circular hole, in uniaxial tension: the
reference solution of the harness exercise (E8, draft).

Plate: width W (along x), length 2 W (along y, loading direction), hole of
diameter d, linear elastic, plane stress, uniform traction sigma on the ends.
By symmetry, a quarter of the plate is meshed (x >= 0, y >= 0), with
u_x = 0 on x = 0 and u_y = 0 on y = 0.

The stress concentration factor is defined with respect to the nominal stress
on the net section: K_t = max sigma_yy / sigma_net, with
sigma_net = sigma W / (W - d). The maximum is at the edge of the hole, at
(d/2, 0).

`kt_heywood(d_over_W)` is Heywood's approximation, K_t = 2 + (1 - d/W)³ (net
section), accurate to about 1% for d/W <= 0.5 when the plate is long; for an
infinite plate (d/W -> 0), K_t -> 3 (Kirsch).

`solve(...)` is a reference finite element solution (scikit-fem, quadratic
triangles on a structured mesh graded towards the hole); it returns a dict with
the quantities that the scripts written by the agent must also return (cf.
`REQUIRED_KEYS`).

Units: N, mm, MPa.
"""

import numpy

REQUIRED_KEYS = ("K_t", "sigma_max", "sigma_net", "reaction_force", "applied_force", "n_dofs")

def kt_heywood(d_over_W):
    """Heywood's approximation of the stress concentration factor (net section)."""
    return 2. + (1. - d_over_W)**3

def mesh(W=50., d=10., n_theta=24, n_radial=24, grading=3.):
    """Structured triangular mesh of the quarter plate (x >= 0, y >= 0) of
    width W, length 2 W, with a hole of diameter d."""
    import skfem
    a, X, Y = d/2, W/2, W # quarter plate: [0, X] x [0, Y] minus the disk of radius a
    theta_c = numpy.arctan2(Y, X)
    n1 = max(2, int(round(n_theta * theta_c / (numpy.pi/2))))
    theta = numpy.concatenate([numpy.linspace(0., theta_c, n1+1), numpy.linspace(theta_c, numpy.pi/2, n_theta-n1+1)[1:]])
    s = (numpy.linspace(0., 1., n_radial+1))**grading # graded towards the hole
    c, sn = numpy.cos(theta), numpy.sin(theta)
    r_out = numpy.minimum(X / numpy.where(c > 1e-12, c, 1e-12), Y / numpy.where(sn > 1e-12, sn, 1e-12)) # ray / rectangle intersection
    P_in = numpy.stack([a*c, a*sn], axis=1)
    P_out = numpy.stack([r_out*c, r_out*sn], axis=1)
    P_out[0] = [X, 0.]; P_out[-1] = [0., Y]
    p = (P_in[None,:,:] * (1 - s[:,None,None]) + P_out[None,:,:] * s[:,None,None]).reshape(-1, 2) # (radial, angular)
    nt = len(theta)
    idx = lambda i, j: i * nt + j
    t = []
    for i in range(n_radial):
        for j in range(nt - 1):
            q = [idx(i, j), idx(i+1, j), idx(i+1, j+1), idx(i, j+1)]
            t += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    m = skfem.MeshTri(p.T, numpy.array(t).T)
    tol = 1e-8 * W
    return m.with_boundaries({
        "left":   lambda x: numpy.abs(x[0]) < tol,
        "bottom": lambda x: numpy.abs(x[1]) < tol,
        "top":    lambda x: numpy.abs(x[1] - Y) < tol,
        "right":  lambda x: numpy.abs(x[0] - X) < tol})

def solve(W=50., d=10., thickness=1., sigma=100., E=210e3, nu=0.3, n_theta=24, n_radial=24, grading=3.):
    """Reference finite element solution (plane stress, quarter model, P2)."""
    import skfem
    from skfem.helpers import dot, ddot, sym_grad, eye, trace
    m = mesh(W, d, n_theta, n_radial, grading)
    e = skfem.ElementVector(skfem.ElementTriP2())
    basis = skfem.Basis(m, e, intorder=4)
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    lam = 2*lam*mu/(lam + 2*mu) # plane stress
    def stress(eps): return 2*mu*eps + lam*eye(trace(eps), 2)
    @skfem.BilinearForm
    def a_form(u, v, w): return thickness * ddot(stress(sym_grad(u)), sym_grad(v))
    fbasis = skfem.FacetBasis(m, e, facets=m.boundaries["top"], intorder=4)
    @skfem.LinearForm
    def l_form(v, w): return thickness * sigma * v[1]
    K = a_form.assemble(basis)
    f = l_form.assemble(fbasis)
    D = numpy.concatenate([basis.get_dofs("left").all("u^1"), basis.get_dofs("bottom").all("u^2")]) # all: vertex and mid-edge dofs
    u = skfem.solve(*skfem.condense(K, f, D=D))
    # Reaction force on the symmetry plane y = 0 (y component), from the residual
    r = K @ u - f
    reaction = -numpy.sum(r[basis.get_dofs("bottom").all("u^2")])
    # sigma_yy at the edge of the hole (d/2, 0): evaluated from the P2 field in the elements that share this node, and averaged
    k = int(numpy.argmin(numpy.sum((m.p.T - [d/2, 0.])**2, axis=1)))
    elems = numpy.flatnonzero(numpy.any(m.t == k, axis=0))
    mapping = m._mapping()
    values = []
    for el in elems:
        X = mapping.invF(m.p[:, k][:, None, None], tind=numpy.array([el]))[:, 0, :]
        b = skfem.Basis(m, e, elements=numpy.array([el]), quadrature=(X, numpy.ones(1)))
        values.append(float(stress(sym_grad(b.interpolate(u)))[1, 1].ravel()[0]))
    sigma_max = float(numpy.mean(values))
    sigma_net = sigma * W / (W - d)
    return dict(K_t=sigma_max/sigma_net, sigma_max=sigma_max, sigma_net=float(sigma_net),
                reaction_force=float(reaction), applied_force=float(sigma * thickness * W/2), n_dofs=int(basis.N))

################################################################################
### Standalone scripts, for testing the verification checks (E8, part C) ######
################################################################################

_SCRIPT = '''#coding=utf8
"""Stress concentration factor of a finite-width plate with a central hole, in
uniaxial tension (plane stress, linear elasticity, quarter model, scikit-fem).
Usage: python plate.py --W 50 --d 10 --n 24 (mm, MPa; --n: mesh resolution).
Prints one line of JSON with K_t, sigma_max, sigma_net, reaction_force, applied_force, n_dofs."""
import argparse, json
import numpy, skfem
from skfem.helpers import ddot, sym_grad, eye, trace

parser = argparse.ArgumentParser()
for name, value in [("W", 50.), ("d", 10.), ("thickness", 1.), ("sigma", 100.), ("E", 210e3), ("nu", 0.3)]:
    parser.add_argument("--"+name, type=float, default=value)
parser.add_argument("--n", type=int, default=24)
args = parser.parse_args()
W, d, t, sigma, E, nu, n = args.W, args.d, args.thickness, args.sigma, args.E, args.nu, @N@

# Mesh of the quarter plate [0, W/2] x [0, W] minus the disk of radius d/2, graded towards the hole
a, X, Y = d/2, W/2, W
theta_c = numpy.arctan2(Y, X)
n1 = max(2, int(round(n * theta_c / (numpy.pi/2))))
theta = numpy.concatenate([numpy.linspace(0., theta_c, n1+1), numpy.linspace(theta_c, numpy.pi/2, n-n1+1)[1:]])
s = numpy.linspace(0., 1., n+1)**3
c, sn = numpy.cos(theta), numpy.sin(theta)
r_out = numpy.minimum(X / numpy.where(c > 1e-12, c, 1e-12), Y / numpy.where(sn > 1e-12, sn, 1e-12))
P_in, P_out = numpy.stack([a*c, a*sn], axis=1), numpy.stack([r_out*c, r_out*sn], axis=1)
P_out[0], P_out[-1] = [X, 0.], [0., Y]
p = (P_in[None] * (1 - s[:,None,None]) + P_out[None] * s[:,None,None]).reshape(-1, 2)
nt = len(theta)
tri = []
for i in range(n):
    for j in range(nt - 1):
        q = [i*nt+j, (i+1)*nt+j, (i+1)*nt+j+1, i*nt+j+1]
        tri += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
tol = 1e-8 * W
m = skfem.MeshTri(p.T, numpy.array(tri).T).with_boundaries({
    "left": lambda x: numpy.abs(x[0]) < tol, "bottom": lambda x: numpy.abs(x[1]) < tol, "top": lambda x: numpy.abs(x[1] - Y) < tol})

# Plane-stress linear elasticity, quadratic elements
e = skfem.ElementVector(skfem.ElementTriP2())
basis = skfem.Basis(m, e, intorder=4)
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
lam = 2*lam*mu/(lam + 2*mu)
stress = lambda eps: 2*mu*eps + lam*eye(trace(eps), 2)
K = skfem.BilinearForm(lambda u, v, w: t * ddot(stress(sym_grad(u)), sym_grad(v))).assemble(basis)
f = skfem.LinearForm(lambda v, w: t * @TRACTION@ * v[1]).assemble(skfem.FacetBasis(m, e, facets=m.boundaries["top"], intorder=4))
D = numpy.concatenate([basis.get_dofs("@LEFT@").@KIND@("u^1"), basis.get_dofs("@BOTTOM@").@KIND@("u^2")])
u = skfem.solve(*skfem.condense(K, f, D=D))
reaction = -numpy.sum((K @ u - f)[basis.get_dofs("@BOTTOM@").@KIND@("u^2")])

# sigma_yy at the edge of the hole (d/2, 0), averaged over the elements that share this node
k = int(numpy.argmin(numpy.sum((m.p.T - [a, 0.])**2, axis=1)))
values = []
for el in numpy.flatnonzero(numpy.any(m.t == k, axis=0)):
    Xr = m._mapping().invF(m.p[:, k][:, None, None], tind=numpy.array([el]))[:, 0, :]
    b = skfem.Basis(m, e, elements=numpy.array([el]), quadrature=(Xr, numpy.ones(1)))
    values.append(float(stress(sym_grad(b.interpolate(u)))[1, 1].ravel()[0]))
sigma_max = float(numpy.mean(values))
sigma_net = @SIGMA_NET@
print(json.dumps(dict(K_t=sigma_max/sigma_net, sigma_max=sigma_max, sigma_net=sigma_net,
                      reaction_force=float(reaction), applied_force=float(@APPLIED@), n_dofs=int(basis.N))))
'''

_VARIANTS = {
    # name: (description, replacements)
    "reference":      ("correct script", {}),
    "gross_section":  ("K_t computed with the gross-section nominal stress", {"@SIGMA_NET@": "sigma"}),
    "fixed_mesh":     ("the mesh resolution argument is ignored (always n = 6)", {"@N@": "6"}),
    "mid_edge_dofs":  ("only the vertex dofs are constrained on the symmetry planes (quadratic elements)", {"@KIND@": "nodal_key"}),
    "wrong_symmetry": ("symmetry conditions swapped (u_x = 0 on y = 0, u_y = 0 on x = 0)", {"@LEFT@": "bottom", "@BOTTOM@": "left"}),
    "full_width":     ("applied force reported for the full width, on a quarter model", {"@APPLIED@": "sigma * t * W"}),
}

def script(variant="reference"):
    """Source code of a standalone script (command-line interface and JSON
    output as required in E8), either correct ("reference") or with a typical
    error (cf. `VARIANTS`), to test verification checks."""
    defaults = {"@N@": "args.n", "@TRACTION@": "sigma", "@LEFT@": "left", "@BOTTOM@": "bottom", "@KIND@": "all", "@SIGMA_NET@": "sigma * W / (W - d)", "@APPLIED@": "sigma * t * W/2"}
    defaults.update(_VARIANTS[variant][1])
    src = _SCRIPT
    for key, value in defaults.items():
        src = src.replace(key, value)
    if (defaults["@KIND@"] == "nodal_key"): # vertex dofs only
        src = src.replace('.nodal_key("u^1")', '.nodal["u^1"]').replace('.nodal_key("u^2")', '.nodal["u^2"]')
    return src

VARIANTS = {name: description for name, (description, _) in _VARIANTS.items()}
