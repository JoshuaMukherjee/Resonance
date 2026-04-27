from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, DTYPE
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points, get_normals_as_points, get_tetra_centroids
from acoustools.BEM import propagate_BEM_pressure, compute_E, compute_A, augment_A_CHIEF, compute_H, propagate_BEM_laplacian_abs
from acoustools.Constants import wavelength,k, P_ref

import os
import pickle

import torch

from torch import Tensor
from vedo import Mesh

import matplotlib.pyplot as plt

board = TOP_BOARD

path = "../BEMMedia/"

p_ref = 12 * 0.22

# scatterer_path = 'blob.stl'
scatterer_path = 'Sphere-lam2.stl'

scatterer = load_scatterer(path + scatterer_path)
centre_scatterer(scatterer)
print(scatterer.bounds())
d = wavelength*2


scale_to_diameter(scatterer,d)
get_edge_data(scatterer)

x = iterative_backpropagation(create_points(1,1,0,0,0), board=board)



com = get_centre_of_mass_as_points(scatterer)


areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
norms = get_normals_as_points(scatterer, permute_to_points=False)

tetra_centres = get_tetra_centroids(scatterer)
N = tetra_centres.shape[2]
print(N)


sample_size = 200
sample = tetra_centres[:,:,torch.randint(0, N, (1,sample_size)).squeeze()]


pth = './Resonance/data/outputs/one-point-H/'




fig = plt.figure()
# ax = fig.add_subplot(projection='3d')




max_p = 1

pxs = []
pys = []
pzs = []
values = []

F = len(os.listdir(pth))

for i,f in enumerate(os.listdir(pth)):
    print(i, end='\r')

    H,p = pickle.load(open(pth+f,'rb'))

    pressure_CHIEF = propagate_BEM_pressure(x,sample, H=H, path=path, scatterer=scatterer, board=board).mean()

    pxs.append(tetra_centres[0,0,i].cpu().detach().real)
    pys.append(tetra_centres[0,1,i].cpu().detach().real)
    pzs.append(tetra_centres[0,2,i].cpu().detach().real)
    values.append((pressure_CHIEF/max_p).item())

# ax.scatter(pxs, pys, pzs, alpha= values)
plt.scatter(range(F),values)

plt.show()

    