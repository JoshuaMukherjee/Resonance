from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A, augment_A_CHIEF
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



Hconds = []
Hconds_CHIEF = []
Hsv = []
Hsv_CHIEF = []

ps = []
ps_CHIEF = []


for i,d in enumerate(ds):

    print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)

    A = compute_A(reflector)
    H = compute_H(reflector, board, A=A)
    min_sv = torch.linalg.svdvals(H).min()
    Hcond = torch.linalg.cond(H)

    E = compute_E(reflector, p, board, H=H)

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    H_CHIEF = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points)
    min_sv_CHIEF = torch.linalg.svdvals(H_CHIEF).min()
    Hcond_CHIEF = torch.linalg.cond(H_CHIEF)
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)



    pressure = torch.abs(E@x)
    pressure_CHIEF = torch.abs(E_CHIEF@x)

    Hconds.append(Hcond.item())
    Hconds_CHIEF.append(Hcond_CHIEF.item())
    
    ps.append(pressure.item())
    ps_CHIEF.append(pressure_CHIEF.item())

    Hsv.append(min_sv.item())
    Hsv_CHIEF.append(min_sv_CHIEF.item())




import matplotlib.pyplot as plt


plt.subplot(4,1,1)
plt.scatter(Hconds, ps, label = 'H')
plt.scatter(Hconds_CHIEF, ps_CHIEF, label = 'H CHIEF')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Cond')
plt.legend()

plt.subplot(4,1,2)
plt.scatter(ds, Hconds, label = 'H')
plt.scatter(ds, Hconds_CHIEF, label = 'H CHIEF')
plt.xlabel('Diameter (m)')
plt.ylabel('Cond')
plt.legend()

plt.subplot(4,1,3)
plt.scatter(Hsv, Hconds,label = 'H')
plt.scatter(Hsv_CHIEF, Hconds_CHIEF, label = 'H CHIEF')
plt.xlabel('Min singular Value')
plt.ylabel('Cond')
plt.legend()

plt.subplot(4,1,4)
plt.scatter(Hsv, ps,label = 'H')
plt.scatter(Hsv_CHIEF, ps_CHIEF, label = 'H CHIEF')
plt.xlabel('Min singular Value')
plt.ylabel('Pressure (Pa)')
plt.legend()




plt.show()