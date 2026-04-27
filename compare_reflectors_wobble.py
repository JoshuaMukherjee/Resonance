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
import pickle

board = TRANSDUCERS
p = create_points(1,1,0,0.0,-0.0)

path = "../BEMMedia"

reflector = load_scatterer(path + '/Wobble-Tunnel-lam4.stl')

d = wavelength*5

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

print(reflector.bounds())
get_edge_data(reflector)

COMPUTE = False

if COMPUTE:
    H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, method='OLS')
    E = compute_E(reflector, p, board, H=H)

    # internal_points  = get_CHIEF_points(reflector, P = 10, start='centre', method='uniform', scale=0.45, scale_mode='diameter-scale')
    P=-1
    internal_points = get_CHIEF_points(reflector, P=P, start='surface', scale=0.001, scale_mode='abs')
    H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points, method='OLS')
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    pickle.dump([H,E,H_CHIEF, E_CHIEF, internal_points], open('./Resonance/data/WT-lam4-objs.bin', 'wb'))
else:
    H,E,H_CHIEF, E_CHIEF, internal_points = pickle.load(open('./Resonance/data/WT-lam4-objs.bin', 'rb'))
    #

def compute_trap(point, Emat,Hmat, baord):

    def min_U(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params):
        U = BEM_gorkov_analytical(transducer_phases, points, reflector, E=Emat, internal_points=None, path=path, board=board, H=Hmat)
        # print(U)
        return U.mean().unsqueeze(0)

    x = gradient_descent_solver(point, min_U,board, lr=1e20, log=True)

    return x


x = compute_trap(p, E, H, board)
xCHIEF = compute_trap(p, E_CHIEF, H_CHIEF, board)
# xCHIEF = iterative_backpropagation(p, board=board, A=E_CHIEF)
# xCHIEF = add_lev_sig(xCHIEF)

# x = iterative_backpropagation(p, board=board, A=E)
# x = add_lev_sig(x)

Visualise(*ABC(0.03, plane='yz'), [x,xCHIEF, x, xCHIEF], res = (100,100), points=p,
        colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical, propagate_BEM_pressure, propagate_BEM_pressure],
        colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {}],
        link_ax=[0,1],
        arrangement=(2,2),
        # cmaps=['hsv','hsv', 'hsv']
        cmaps=['seismic','seismic', 'hot', 'hot']
        )

