from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, get_cache_or_compute_H, BEM_gorkov_analytical, BEM_compute_force
from acoustools.Constants import wavelength,k
from acoustools.Paths import interpolate_circle, distance
from acoustools.Solvers import gradient_descent_solver
from torch import Tensor
import torch
import os
import matplotlib.pyplot as plt

board = TRANSDUCERS
p = create_points(1,1,0,0.0,-0.02)

path = "../BEMMedia"

reflector = load_scatterer(path + '/Sphere-lam2.stl')

d = wavelength*3

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

com = get_centre_of_mass_as_points(reflector)

print(reflector.bounds())

H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, method='OLS')
E = compute_E(reflector, p, board, H=H)
x = iterative_backpropagation(p, board=board, A=E)

internal_points  = get_CHIEF_points(reflector, P = 50, start='centre', method='uniform', scale=0.1, scale_mode='diameter-scale')
H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points, method='OLS')
E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)
#

N = 100

Fx_chief = []
Fx = []

Fy_chief = []
Fy = []

Fz_chief = []
Fz = []

for i in range(N):
    print(i, end='\r')

    sampled = False
    while not sampled:
        pt = create_points(1,1)
        if distance(pt, com) < d/2:
            sampled = True
    
    force = BEM_compute_force(x, pt, board, H=H, path=path, scatterer=reflector).squeeze()
    
    fx= force[0].item()
    fy= force[1].item()
    fz= force[2].item()

    Fx.append(fx)
    Fy.append(fy)
    Fz.append(fz)

    force_chief = BEM_compute_force(x, pt, board, H=H_CHIEF, path=path, scatterer=reflector).squeeze()

    
    fx_chief= force_chief[0].item()
    fy_chief= force_chief[1].item()
    fz_chief= force_chief[2].item()

    Fx_chief.append(fx_chief)
    Fy_chief.append(fy_chief)
    Fz_chief.append(fz_chief)
    

plt.scatter(Fx_chief, Fx)
plt.scatter(Fy_chief, Fy)
plt.scatter(Fz_chief, Fz)

plt.show()