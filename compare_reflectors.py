from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, get_cache_or_compute_H, BEM_gorkov_analytical, BEM_compute_force
from acoustools.Constants import wavelength,k
from acoustools.Paths import interpolate_circle
from acoustools.Solvers import gradient_descent_solver
from torch import Tensor
import torch
import os

board = TRANSDUCERS
p = create_points(1,1,0,0.0,-0.02)

path = "../BEMMedia"

reflector = load_scatterer(path + '/Sphere-lam2.stl')

d = wavelength*3

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

print(reflector.bounds())

H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, method='OLS')
E = compute_E(reflector, p, board, H=H)
x = iterative_backpropagation(p, board=board, A=E)

internal_points  = get_CHIEF_points(reflector, P = 50, start='centre', method='uniform', scale=0.1, scale_mode='diameter-scale')
P=80
# internal_points = torch.cat(interpolate_circle(p, radius=(d/2)*0.9, n=P, plane='xz'), dim=2) 
H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points, method='OLS')
E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)
#

def compute_trap(point, E, baord):

    def min_U(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params):
        U = BEM_gorkov_analytical(transducer_phases, points, reflector, E=E, internal_points=None, H=H, path=path)
        return U.mean().unsqueeze(0)

    x = gradient_descent_solver(point, min_U,board, lr=1e3, log=False, iters=500)

    return x

def compute_fz(activations, points, board, H, **params):
    force = BEM_compute_force(activations, points, board, H=H, path=path, scatterer=reflector).squeeze()
    return force[2].unsqueeze(0)

x = compute_trap(p, E, board)
# xCHIEF = compute_trap(p, E_CHIEF, board)
# iterations = 1000
# xCHIEF = iterative_backpropagation(p, board=board, A=E_CHIEF, iterations=iterations)
# xCHIEF = add_lev_sig(xCHIEF)
# 
# x = iterative_backpropagation(p, board=board, A=E, iterations=iterations)
# x = add_lev_sig(x)


Visualise(*ABC(0.0003, plane='xz', origin=p), [x,x,x,x,x,x], res = (100,100), points=p,
        colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical, propagate_BEM_pressure, propagate_BEM_pressure, compute_fz, compute_fz ],
        colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},

                                {}],
        # link_ax=[4,5],
        link_ax=None,
        arrangement=(3,2),
        # cmaps=['hsv','hsv', 'hsv']
        # vmax = [0,0,6000,6000],
        # vmin= [-5e-7, -5e-7, 0,0],
        
        cmaps=['seismic','seismic', 'hot', 'hot', 'berlin', 'berlin'],
        show=False
        )

