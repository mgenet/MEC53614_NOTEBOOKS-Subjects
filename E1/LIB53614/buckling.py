#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
An imperfection-sensitive structure: a rigid bar of length L, pinned at its
base, held at its top by a lateral spring that softens with the deflection,
and loaded by an axial force P. The bar has an initial imperfection (tilt
angle theta_0). In terms of the load parameter lambda = P / P_c, where P_c is
the critical load of the perfect structure, the equilibrium path reads
    lambda(theta) = cos(theta) (1 - sin(theta_0) / sin(theta)) (1 - beta (sin(theta) - sin(theta_0))²),
with beta the softening coefficient of the spring. The perfect structure
(theta_0 = 0) has a symmetric unstable bifurcation at lambda = 1; the imperfect
one reaches a limit load lambda_max < 1, which `limit_load` computes by
following the equilibrium path.

Each call to `limit_load` plays the role of an expensive nonlinear simulation,
and is counted in `n_calls`.
"""

import numpy

n_calls = 0

def limit_load(theta_0, beta=2.):
    """Limit load lambda_max (normalized by the critical load of the perfect
    structure) for the imperfection theta_0 (rad, scalar). Each call counts
    as one simulation."""
    global n_calls
    n_calls += 1
    theta_0 = float(theta_0)
    if (theta_0 <= 0.): return 1.
    s_0 = numpy.sin(theta_0)
    # Path following: theta from theta_0 (lambda = 0) up to the limit point
    theta = numpy.linspace(theta_0, numpy.pi/2, 20001)[1:]
    s = numpy.sin(theta)
    lmbda = numpy.cos(theta) * (1. - s_0/s) * (1. - beta * (s - s_0)**2)
    k = numpy.argmax(lmbda)
    # Local refinement around the maximum
    a, b = theta[max(k-1, 0)], theta[min(k+1, len(theta)-1)]
    theta = numpy.linspace(a, b, 2001)
    s = numpy.sin(theta)
    lmbda = numpy.cos(theta) * (1. - s_0/s) * (1. - beta * (s - s_0)**2)
    return float(numpy.max(lmbda))

def reset_counter():
    global n_calls
    n_calls = 0
