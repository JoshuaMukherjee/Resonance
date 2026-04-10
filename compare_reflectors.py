from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, get_cache_or_compute_H, BEM_gorkov_analytical, BEM_compute_force
from acoustools.Constants import wavelength,k

import torch
import os

board = TOP_BOARD
p = create_points(1,1,0,0,0.01)

path = "../BEMMedia"

reflector = load_scatterer(path + '/sphere-lam2.stl')
scale_to_diameter(reflector, wavelength*2)
centre_scatterer(reflector)

H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False)
E = compute_E(reflector, p, board, H=H)
x = iterative_backpropagation(p, board=board, A=E)

internal_points  = get_CHIEF_points(reflector, P = 20, start='centre', method='uniform')
H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points)
E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)
x_CHIEF = iterative_backpropagation(p, board=board, A=E_CHIEF)

def compute_force_z(activations,points,board, H=None,scatterer=None, path=None):
    force = BEM_compute_force(activations,points,board, H=H, scatterer=scatterer, path=path)[2,:].unsqueeze(0)
    return force

Visualise(*ABC(0.02), [x,x_CHIEF], res = (100,100),
        colour_functions=[compute_force_z, compute_force_z,'-'],
        colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H},
                                {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                {}],
        # link_ax=[0,1],
        )