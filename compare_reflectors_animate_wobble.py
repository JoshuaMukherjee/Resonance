from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, propagate_abs, TRANSDUCERS, BOARD_POSITIONS
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, propagate_BEM_phase, get_cache_or_compute_H, BEM_gorkov_analytical, BEM_compute_force
from acoustools.Constants import wavelength,k
from acoustools.Paths import interpolate_circle, interpolate_points
from acoustools.Solvers import gradient_descent_solver, gorkov_target
from acoustools.Export.Holo import save_holograms, load_holograms
from acoustools.Export.CSV import write_to_file
from torch import Tensor
import torch, pickle, vedo
import os
import matplotlib.pyplot as plt

board = TOP_BOARD

path = "../BEMMedia"

reflector = load_scatterer(path + '/LargeTunnel-full-lam2.stl')
# reflector = load_scatterer(path + '/Sphere-lam1.stl')
# d = wavelength*5
bounds = reflector.bounds()
scale_to_diameter(reflector, (bounds[1] - bounds[0])/1000)
# vedo.show(reflector, axes=1)

print(BOARD_POSITIONS)
print(reflector.bounds())
# get_edge_data(reflector)

# vedo.show(reflector, axes=1)
# exit()

COMPUTE = False

holo = torch.zeros(1,256,1).to(device=device)
holo_CHIEF = torch.zeros(1,256,1).to(device=device)

if COMPUTE:
    H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, method='OLS')
    # E = compute_E(reflector, p, board, H=H)

    # internal_points  = get_CHIEF_points(reflector, P = 10, start='centre', method='uniform', scale=0.45, scale_mode='diameter-scale')
    internal_points = get_CHIEF_points(reflector, P=-1, start='tetra-random')
    H_CHIEF = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False, internal_points=internal_points, method='OLS')
    # E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    pickle.dump([H,H_CHIEF, internal_points], open('./Resonance/data/WT-lam2-objs.bin', 'wb'))
    # pickle.dump([H,H_CHIEF, internal_points], open('./data/WT-lam2-objs.bin', 'wb'))
else:
    H,H_CHIEF, internal_points = pickle.load(open('./Resonance/data/WT-lam2-objs.bin', 'rb'))
    # H,H_CHIEF, internal_points = pickle.load(open('./data/WT-lam2-objs.bin', 'rb'))

# exit()
start = create_points(1,1,0,-0.03,0.03)
end = create_points(1,1,0,0.03,0.03)
path = interpolate_points(start, end, n=15000)


# origin = create_points(0,0,0,0,0)
# path = interpolate_circle(origin, 0.01, n = 100)


prev_U  = [None, None]

def compute_trap(point,Hmat, baord, start, prev, lr):

    def min_U(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params):
        U = BEM_gorkov_analytical(transducer_phases, points, reflector, internal_points=None, path=path, board=board, H=Hmat, dims='Z')
        # print(U)
        return U.mean().unsqueeze(0)
    
    def U_change(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, prev=None, **objective_params):
        U = BEM_gorkov_analytical(transducer_phases, points, reflector, internal_points=None, path=path, board=board, H=Hmat, dims='Z')
        if prev_U is not None:
            dU = prev_U[prev].squeeze() - U.squeeze()
            prev_U[prev] = U.clone().detach()

        
        return dU.abs().unsqueeze(0)


    def phase_change(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params):
        return (transducer_phases.angle() - targets.angle()).abs().mean()

    def objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, prev=None, **objective_params):
        if prev_U[prev] is None:
            U = min_U(transducer_phases, points, board, targets) + 1e-7 * phase_change(transducer_phases, points, board, targets)
            # prev_U[prev] = U.clone().detach()
            return U
        # else:
        #     return U_change(transducer_phases, points, board, targets, prev=prev) + 1e-7 * phase_change(transducer_phases, points, board, targets)

   
    x = gradient_descent_solver(point, objective,board, lr=lr, log=False, targets=start, objective_params={'prev':prev})

    return x


for n,p in enumerate(path):

    # print(n, end='\r')
    

    x = compute_trap(p, H, board, start = holo, prev = 0, lr=1e0)
    xCHIEF = compute_trap(p, H_CHIEF, board, start=holo_CHIEF, prev=1, lr=1e10)

    print(BEM_gorkov_analytical(x, p, H=H, board=board, path=path, scatterer=reflector).item(), 
          BEM_gorkov_analytical(x, p, H=H_CHIEF, board=board, path=path,scatterer=reflector).item(), 
          BEM_gorkov_analytical(xCHIEF, p, H=H_CHIEF, board=board, path=path, scatterer=reflector).item())

    # if holo is not None: print(n,(holo.angle() - x.angle()).abs().mean().item(), (holo_CHIEF.angle() - xCHIEF.angle()).abs().mean().item())
    print(n, x.sum(), xCHIEF.sum())

    # print('Visualising')
    # Visualise(*ABC(0.07, plane='yz'), [x,xCHIEF], res = (150,150), points=p,

    #         colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical],
    #         colour_function_args=[{'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
    #                                 {'path':path, 'board':board, 'scatterer':reflector, "H":H_CHIEF},
                                   
    #                                 {}],
    #         link_ax=[0,1],
    #         # cmaps=['hsv','hsv', 'hsv']
    #         cmaps=['seismic', 'seismic']
    #         )
    # exit()

    # save_holograms(x, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/BEM/{n}.holo')
    # save_holograms(xCHIEF, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/CHIEF/{n}.holo')

    # xload = load_holograms(f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/BEM/{n}.holo')[0]
    # xCHIEFload = load_holograms(f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/CHIEF/{n}.holo')[0]

    # print(n, xload.sum(), xCHIEFload.sum())

    write_to_file(x, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/BEM/{n}.csv', 1, num_transducers=256, flip=False)
    write_to_file(xCHIEF, f'./Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/CHIEf/{n}.csv', 1, num_transducers=256, flip=False)

    # if holo is not None: print(n,(holo.angle() - xload.angle()).abs().mean(), (holo_CHIEF.angle() - xCHIEFload.angle()).abs().mean())

    # print(BEM_gorkov_analytical(xload, p, H=H, board=board, path=path, scatterer=reflector), 
    #       BEM_gorkov_analytical(xload, p, H=H_CHIEF, board=board, path=path,scatterer=reflector), 
    #       BEM_gorkov_analytical(xCHIEFload, p, H=H_CHIEF, board=board, path=path, scatterer=reflector))
    # holo_CHIEF = xCHIEF.clone()

    # #
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