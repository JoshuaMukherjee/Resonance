from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points
from acoustools.Utilities import random_points_ring, TRANSDUCERS, create_points
from acoustools.Constants import wavelength, k as k_c
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_forward_model_second_derivative_unmixed, BEM_laplacian, propagate_BEM_laplacian_abs, get_cache_or_compute_H
from acoustools.Solvers import kd_solver

import torch

path = '../BEMMedia/'
scatterer = load_scatterer(path + 'Sphere-lam2.stl')

board = TRANSDUCERS


d = wavelength *2
centre_scatterer(scatterer)
scale_to_diameter(scatterer, diameter=d)

points = random_points_ring(10000, min_r=d + 1*wavelength)

internal_points  = get_CHIEF_points(scatterer, P = 50, method='tetra-random')

H_CHIEF = get_cache_or_compute_H(scatterer, board, path=path, internal_points=internal_points, use_cache_H=False)
H = get_cache_or_compute_H(scatterer, board, path=path, internal_points=None, use_cache_H=False)

E_CHIEF_lap = BEM_laplacian(points, scatterer, board, use_cache_H=False, H=H_CHIEF)
E_lap = BEM_laplacian(points, scatterer, board, use_cache_H=False, H=H)

E_CHIEF = compute_E(scatterer, points, board, internal_points=internal_points, H=H_CHIEF, use_cache_H=False)
E = compute_E(scatterer, points, board, internal_points=internal_points, H=H, use_cache_H=False)


x = kd_solver(create_points(1,1,0,0,0), board)
# x = torch.exp(torch.rand_like(x) * 2 * 3.14 * 1j)


lap_CHIEF = E_CHIEF_lap @ x
lap = E_lap @ x

p_CHIEF = E_CHIEF @ x
p = E @ x


k_CHIEF = (((lap_CHIEF / p_CHIEF) * -1) ** 0.5).detach().squeeze()
k = (((lap / p) * -1) ** 0.5).detach().squeeze()

print(k_c)
print(k_CHIEF.mean())
print(k.mean())

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k.real, k.imag, label='BEM')
plt.scatter(k_CHIEF.real, k_CHIEF.imag, label='CHIEF')
plt.vlines(k_c, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], linestyle=':')
plt.hlines(0, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], linestyle=':')
plt.ylabel("Im(k)")
plt.xlabel("Re(k)")
plt.legend()



plt.subplot(1,2,2)
BINS =500
plt.hist([k.real/k_c, k_CHIEF.real/k_c], bins=BINS, alpha=1, label=['BEM','CHIEF'])
plt.xlabel("Re(k)")
plt.ylabel("Frequency")
plt.legend()
plt.xlim(0.995, 1.005)

plt.show()
