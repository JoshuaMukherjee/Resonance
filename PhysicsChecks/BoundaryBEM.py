from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points, get_normals_as_points, get_centres_as_points, get_edge_data
from acoustools.Utilities import random_points_ring, TRANSDUCERS, create_points, propagate
from acoustools.Constants import wavelength, k as k_c
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_forward_model_second_derivative_unmixed, BEM_laplacian, propagate_BEM_laplacian_abs, get_cache_or_compute_H, compute_A
from acoustools.Solvers import kd_solver

from acoustools.Visualiser import Visualise, ABC

import torch, vedo
import numpy as np

path = '../BEMMedia/'
scatterer = load_scatterer(path + 'Sphere-lam1.stl')

board = TRANSDUCERS


# d = wavelength * 2
d = wavelength * 2.5
centre_scatterer(scatterer)
scale_to_diameter(scatterer, diameter=d)
get_edge_data(scatterer)

IP  = get_CHIEF_points(scatterer, P = 50, method='tetra-random')



# vedo.show((scatterer.wireframe(), scatterer_minus.wireframe(), scatterer_plus.wireframe()), axes=1)
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False
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

    x = kd_solver(create_points(1,1,0,0,0), board)
    
    boundary_pressure = H@x
    M = H.shape[1]
    boundary_pressure_exp = boundary_pressure.squeeze(0).expand((M,M)).clone()
    
    
    
    boundary_pressure_exp[torch.eye(M).to(bool)] = 0
    boundary_pressure_exp.unsqueeze_(0)
    A= -1*compute_A(scatterer)

    boundary_pressure_exp = boundary_pressure_exp * A.mT
    # print(boundary_pressure_exp.shape)
    
    boundary_contibutions = boundary_pressure_exp.sum(dim=1).unsqueeze(0)
    # print(boundary_contibutions.shape)
    
    p_inc = propagate(x, get_centres_as_points(scatterer), board).unsqueeze_(0)
    # print(p_inc.shape)
    # exit()
    
    
    total_p = p_inc + boundary_contibutions
    boundary_pressure = boundary_pressure.mT
    # total_p.unsqueeze_(1)
    # print(total_p.shape, boundary_pressure.shape)
    # exit()
    
    res = ((total_p / boundary_pressure))
    # print(res.shape)
    print(res.mean())
    
    # plt.scatter(res.real, res.imag)
    plt.scatter(total_p.abs(), boundary_pressure.abs(), label=labels[j])
    plt.legend()

x = [i for i in range(int(plt.gca().get_xlim()[1]))]
y = [2 * i for i in x]
plt.plot(x,y)

plt.show()
    
    
    
