from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation

import torch

board = TRANSDUCERS
p = create_points(1,1,0,0,-0.02)

path = "../BEMMedia"

reflector = load_scatterer(path + '/sphere-lam2.stl')

d = wavelength*2

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

centres = get_tetra_centroids(reflector)
N = centres.shape[2]
sample_size = 200
print(centres.shape)
sample = centres[:,:,torch.randint(0, N, (1,sample_size)).squeeze()]

H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False)
E = compute_E(reflector, p, board, H=H)
x = iterative_backpropagation(p, board=board, A=E)

pressure = propagate_BEM_pressure(x, sample, H=H, path=path, scatterer=reflector, board=board)
print(pressure.mean())


import matplotlib.pyplot as plt

Ps = []
pressures = []

for i in range(30,70):

    P = i 
    Ps.append(P)

    internal_points  = get_CHIEF_points(reflector, P = P, start='centre')
    H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points)
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)
    x_CHIEF = iterative_backpropagation(p, board=board, A=E_CHIEF)

    pressure_CHIEF = propagate_BEM_pressure(x_CHIEF, sample, H=H_CHIEF, path=path, scatterer=reflector, board=board)
    pressures.append(pressure_CHIEF.mean().item())

plt.plot(Ps,pressures)
plt.yscale('log')

plt.show()