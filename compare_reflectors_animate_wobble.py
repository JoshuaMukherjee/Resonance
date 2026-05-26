from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, get_cache_or_compute_H, BEM_gorkov_analytical, BEM_compute_force
from acoustools.Constants import wavelength,k
from acoustools.Paths import interpolate_circle, interpolate_points
from acoustools.Solvers import gradient_descent_solver
from acoustools.Export.Holo import save_holograms
from torch import Tensor
import torch, pickle, vedo
import os
import matplotlib.pyplot as plt

board = TOP_BOARD

path = "../BEMMedia"

reflector = load_scatterer(path + '/LargeTunnel-varied.stl')
# d = wavelength*5
bounds = reflector.bounds()
scale_to_diameter(reflector, (bounds[1] - bounds[0])/1000)
# vedo.show(reflector, axes=1)


print(reflector.bounds())
# get_edge_data(reflector)

# vedo.show(reflector, axes=1)
# exit()

COMPUTE = True

if COMPUTE:
    H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, method='OLS')
    # E = compute_E(reflector, p, board, H=H)

    # internal_points  = get_CHIEF_points(reflector, P = 10, start='centre', method='uniform', scale=0.45, scale_mode='diameter-scale')
    internal_points = get_CHIEF_points(reflector, P=-1, start='tetra-random')
    H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points, method='OLS')
    # E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    # pickle.dump([H,E,H_CHIEF, E_CHIEF, internal_points], open('./Resonance/data/WT-lam4-objs.bin', 'wb'))
else:
    H,E,H_CHIEF, E_CHIEF, internal_points = pickle.load(open('./Resonance/data/WT-lam4-objs.bin', 'rb'))


start = create_points(1,1,0,0.04,0.03)
end = create_points(1,1,0,-0.04,0.03)

path = interpolate_points(start, end, n=1000)


def compute_trap(point,Hmat, baord):

    def min_U(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params):
        U = BEM_gorkov_analytical(transducer_phases, points, reflector, internal_points=None, path=path, board=board, H=Hmat, dims='Z')
        # print(U)
        return U.mean().unsqueeze(0)

    x = gradient_descent_solver(point, min_U,board, lr=1e20, log=False)

    return x


for n,p in enumerate(path):
    print(n, end='\r')
    

    x = compute_trap(p, H, board)
    xCHIEF = compute_trap(p, H_CHIEF, board)

    # print('Visualising')
    # Visualise(*ABC(0.07, plane='yz'), [x,xCHIEF, x, xCHIEF], res = (200,200),
    #         colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical, propagate_BEM_pressure, propagate_BEM_pressure],
    #         colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {}],
    #         link_ax=[0,1],
    #         arrangement=(2,2),
    #         # cmaps=['hsv','hsv', 'hsv']
    #         cmaps=['seismic','seismic', 'hot', 'hot']
    #         )


    save_holograms(x, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos/BEM/{n}.holo')
    save_holograms(xCHIEF, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos/CHIEF/{n}.holo')

    # exit()

    # plt.gcf().clear()
    # # plt.gca().clear()

    # Visualise(*ABC(0.062, plane='yz'), [x,xCHIEF, x, xCHIEF], res = (100,100), points=p,
    #         colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical,propagate_BEM_pressure, propagate_BEM_pressure],
    #         colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {}],
    #         link_ax=[[0,1],[2,3]],
    #         arrangement=(2,2),
    #         # cmaps=['hsv','hsv', 'hsv']
    #         cmaps=['seismic','seismic', 'hot', 'hot'],
    #         show=False,
    #         vmin=-1e-7
    #         )



    # plt.savefig(f'./Resonance/data/compare_reflector_wobble/Z/img{n}.png')