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



Bconds = []
Bconds_CHIEF = []

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
    Bcond = torch.linalg.cond(B)
    
    E = compute_E(reflector, p, board, H=H)

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    B_CHIEF = compute_bs(reflector, board, internal_points=internal_points)
    H_CHIEF = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, bs=B_CHIEF)
    Bcond_CHIEF = torch.linalg.cond(B_CHIEF)
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    print(Bcond, Bcond_CHIEF)

    pressure = torch.abs(E@x)
    pressure_CHIEF = torch.abs(E_CHIEF@x)

    Bconds.append(Bcond.item())
    Bconds_CHIEF.append(Bcond_CHIEF.item())
    
    ps.append(pressure.item())
    ps_CHIEF.append(pressure_CHIEF.item())





import matplotlib.pyplot as plt


plt.subplot(2,1,1)
plt.scatter(Bconds, ps, label = 'B')
plt.scatter(Bconds_CHIEF, ps_CHIEF, label = 'B CHIEF')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Cond')
plt.legend()

plt.subplot(2,1,2)
plt.scatter(ds, Bconds, label = 'B')
plt.scatter(ds, Bconds_CHIEF, label = 'B CHIEF')
plt.xlabel('Diameter (m)')
plt.ylabel('Cond')
plt.legend()



plt.show()