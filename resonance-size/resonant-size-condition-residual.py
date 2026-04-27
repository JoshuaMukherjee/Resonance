from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A, augment_A_CHIEF, compute_bs
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt


board = TRANSDUCERS
p = create_points(1,1,0,0,-0.002)

path = "../BEMMedia"

x = iterative_backpropagation(p, board=board)

ds = [0.01+wavelength * i/40 for i in range(10, 100)]
# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]



residuals = []
CHIEF_residuals = []

ps = []
ps_CHIEF = []


for i,d in enumerate(ds):

    # print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)

    A = compute_A(reflector)
    B = compute_bs(reflector, board)
    H = compute_H(reflector, board, A=A, bs=B)
    E = compute_E(reflector, p, board, H=H)

    residual = A@H - B

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    B_CHIEF = compute_bs(reflector, board, internal_points=internal_points)
    H_CHIEF,_,_,H_full = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, bs=B_CHIEF, return_components=True)
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    chief_residual = A_CHIEF @ H_full - B_CHIEF

    residuals.append(residual.sum().item())
    CHIEF_residuals.append(chief_residual.sum().item())

    pressure = torch.abs(E@x)
    pressure_CHIEF = torch.abs(E_CHIEF@x)

    ps.append(pressure.item())
    ps_CHIEF.append(pressure_CHIEF.item())





import matplotlib.pyplot as plt



plt.subplot(2,1,1)
plt.scatter(residuals, ps, label = 'B')
plt.scatter(CHIEF_residuals, ps_CHIEF, label = 'B CHIEF')
plt.ylabel('Pressure (Pa)')
plt.xlabel('AH-B')
plt.legend()

plt.subplot(2,1,2)
plt.scatter(ds, residuals, label = 'B')
plt.scatter(ds, CHIEF_residuals, label = 'B CHIEF')
plt.xlabel('Diameter (m)')
plt.ylabel('AH-B')
plt.legend()



plt.show()