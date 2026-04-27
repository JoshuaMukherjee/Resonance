from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, DTYPE
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points, get_normals_as_points, get_tetra_centroids
from acoustools.BEM import propagate_BEM_pressure, compute_E, compute_A, augment_A_CHIEF, compute_H, propagate_BEM_laplacian_abs
from acoustools.Constants import wavelength,k, P_ref

import pickle

import torch

from torch import Tensor
from vedo import Mesh

board = TOP_BOARD

path = "../BEMMedia/"
# paths = [path+"/Sphere-lam2.stl"]   
# scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

p_ref = 12 * 0.22

# scatterer_path = 'blob.stl'
scatterer_path = 'Sphere-lam2.stl'

scatterer = load_scatterer(path + scatterer_path)
centre_scatterer(scatterer)
print(scatterer.bounds())
d = wavelength*2

# d = wavelength * 2
# d = wavelength * 1.2345
# d = 0.0234ß
# d = wavelength+0.001
scale_to_diameter(scatterer,d)
get_edge_data(scatterer)


com = get_centre_of_mass_as_points(scatterer)


areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
norms = get_normals_as_points(scatterer, permute_to_points=False)

tetra_centres = get_tetra_centroids(scatterer)
N = tetra_centres.shape[2]
print(N)

startA = compute_A(scatterer, k=k)

for i in range(N):

    print(i, end='\r')

    CHIEF_point = tetra_centres[:,:,i].unsqueeze(2)

    A = augment_A_CHIEF(startA.clone(), internal_points=CHIEF_point, k=k, scatterer=scatterer, centres=centres, areas=areas, norms=norms)
    H = compute_H(scatterer, board, p_ref=p_ref, k=k, A=A, internal_points=CHIEF_point, use_LU=False, use_OLS=True)

    pickle.dump([H, CHIEF_point], open(f'./Resonance/data/outputs/one-point-H/{i}.bin', 'wb'))
