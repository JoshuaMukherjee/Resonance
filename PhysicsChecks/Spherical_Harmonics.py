from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points
from acoustools.Utilities import random_points_ring, BOTTOM_BOARD, create_points
from acoustools.Constants import wavelength, k as k_c
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_forward_model_second_derivative_unmixed, BEM_laplacian, propagate_BEM_laplacian_abs, get_cache_or_compute_H
from acoustools.Solvers import kd_solver

import scipy

from acoustools.SphericalHarmonics.Propagators import spherical_harmonics_propagate

import matplotlib.pyplot as plt

import torch

path = '../BEMMedia/'
scatterer = load_scatterer(path + 'Sphere-lam2.stl')

board = BOTTOM_BOARD


d = wavelength*2
centre_scatterer(scatterer)
scale_to_diameter(scatterer, diameter=d)

internal_points  = get_CHIEF_points(scatterer, P = 50, method='tetra-random')

points = random_points_ring(1000, min_r=d + 0.1*wavelength)


H = get_cache_or_compute_H(scatterer, board, use_cache_H=False, internal_points=internal_points)


com = create_points(1,1,0,0,0)
x = kd_solver(create_points(1,1,0,0,0), board)



N = 85
pressure_sh = spherical_harmonics_propagate(x, points, board=board, centre_of_mass=com, max_n = N, particle_radius=d/2).abs().flatten().tolist()

pressure_BEM = propagate_BEM_pressure(x, points, scatterer, board, path=path).flatten().tolist()
pressure_CHIEF = propagate_BEM_pressure(x, points, scatterer, board, H=H, path=path).flatten().tolist()

slope, intercept, r_value_BEM, p_value_BEM, std_err = scipy.stats.linregress(pressure_sh, pressure_BEM)
slope, intercept, r_value_CHIEF, p_value_CHIEF, std_err = scipy.stats.linregress(pressure_sh, pressure_CHIEF)


plt.scatter(pressure_sh, pressure_BEM, label=f'Standard, $r^2$={round(r_value_BEM**2,2)}')
plt.scatter(pressure_sh, pressure_CHIEF, label=f'CHIEF, $r^2$={round(r_value_CHIEF**2,2)}')
plt.plot(sorted(pressure_sh), sorted(pressure_sh), label='y=x')
plt.legend()
plt.xlabel("Spherical Harmonics Pressure (Pa)")
plt.ylabel('BEM Pressure (Pa)')

plt.show()