from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points, get_normals_as_points, get_centres_as_points, get_edge_data
from acoustools.Utilities import random_points_ring, TRANSDUCERS, create_points
from acoustools.Constants import wavelength, k as k_c
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_forward_model_second_derivative_unmixed, BEM_laplacian, propagate_BEM_laplacian_abs, get_cache_or_compute_H
from acoustools.Solvers import kd_solver

from acoustools.Visualiser import Visualise, ABC

import torch, vedo
import numpy as np

path = '../BEMMedia/'
scatterer = load_scatterer(path + 'Sphere-lam2.stl')

board = TRANSDUCERS


d = wavelength * 2
# d = wavelength * 2.5
centre_scatterer(scatterer)
scale_to_diameter(scatterer, diameter=d)
get_edge_data(scatterer)

IP  = get_CHIEF_points(scatterer, P = 50, method='tetra-random')


fin_diff_h = 1e-6

scatterer_plus = load_scatterer(path + 'Sphere-lam2.stl')
centre_scatterer(scatterer_plus)

scatterer_minus = load_scatterer(path + 'Sphere-lam2.stl')
centre_scatterer(scatterer_minus)
scale_to_diameter(scatterer_minus,diameter = d - 2*fin_diff_h)
scale_to_diameter(scatterer_plus, diameter= d + 2*fin_diff_h)



# vedo.show((scatterer.wireframe(), scatterer_minus.wireframe(), scatterer_plus.wireframe()), axes=1)
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 30}
plt.rc('font', **font)  # pass in the font dict as kwargs

BINS =50

results = []
results_real = []
results_imag = []

labels = ['BEM','CHIEF']

for j,internal_points in enumerate([None, IP]):

    H = get_cache_or_compute_H(scatterer, board, use_cache_H=False, internal_points=internal_points)
    H_plus = get_cache_or_compute_H(scatterer_plus, board, use_cache_H=False, internal_points=internal_points)
    H_minus = get_cache_or_compute_H(scatterer_minus, board, use_cache_H=False, internal_points=internal_points)

    H_grad = (H_plus - H_minus) / (2*fin_diff_h)
    x = kd_solver(create_points(1,1,0,0,0), board)
    
    # Visualise(*ABC(0.04), x, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board, 'path':path, 'scatterer':scatterer, 'H':H}] )

    
    imps = []
    imps_real = []
    imps_imag  = []
    for i in range(1):
        
        # x = torch.exp(torch.rand_like(x) * 2 * 3.14 * 1j)
# 
        p_grad = (H_grad @ x)[0,:,0]
        p = (H @ x)[0,:,0]
        
        imp = p_grad / (1j * k_c * p)
        # print(imp.shape)
        
        imps += imp.abs().tolist()
        imps_real += imp.real.tolist()
        imps_imag += imp.imag.tolist()

    
    print(sum(imps_real) / len(imps_real), sum(imps_imag) / len(imps_imag))
    print(np.median(imps_real), np.median(imps_imag))
    results.append(imps)
    results_real.append(imps_real)
    results_imag.append(imps_imag)
    print()
    
    
    
    plt.subplot(1,2,1)
    plt.scatter(imps_real, imps_imag, label=labels[j])
    plt.xlabel(r"$Re(\beta)$")
    plt.ylabel(r"$Im(\beta)$")
    plt.legend()

# plt.scatter(results_imag,re)
plt.subplot(1,2,2)
plt.hist(results, bins=BINS, alpha=1, label = labels)
plt.ylabel("Frequency")
plt.xlabel(r"$\beta$")
    # plt.plot(sorted(imps))

# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.axis("equal")

plt.legend()


plt.show()
