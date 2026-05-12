from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points
from acoustools.Constants import wavelength
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A, augment_A_CHIEF
from acoustools.Solvers import iterative_backpropagation
from acoustools.Constants import k

import torch
import matplotlib.pyplot as plt
import scipy
import math

board = TRANSDUCERS
p = create_points(1,1,0,0,-0.002)

path = "../BEMMedia"

x = iterative_backpropagation(p, board=board)

N = 50
ds = [0.01 + (0.02) * i/N for i in range(N)]
# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]o



Aeigs = []
Aeigs_CHIEF = []


Aeigs_real = []
Aeigs_CHIEF_real = []

Aeigs_abs = []
Aeigs_CHIEF_abs = []

ps = []
ps_CHIEF = []

ps_eig = []
ps_eig_CHIEF = []


for i,d in enumerate(ds):

    print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)
    p = get_tetra_centroids(reflector)


    A = compute_A(reflector)
    
    
  
    H = compute_H(reflector, board, A=A, use_LU=True, use_OLS=False)
    
    Aeig = torch.linalg.eigvals(A)
    Aeig_abs = Aeig.abs()
    Aeig_abs_sorted ,Aeig_idx = torch.sort(Aeig_abs)
    Aeig_sorted = Aeig[:,Aeig_idx.squeeze()]
    Aeigs_abs.append(Aeig_abs_sorted[:,-1])
    
    Heig = H * Aeig_sorted[:, -1].item()
    
    # print(Aeig_abs_sorted)
    # exit()

    E = compute_E(reflector, p, board, H=H)
    Eeig = compute_E(reflector, p, board, H=Heig)

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    H_CHIEF = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, use_LU=True, use_OLS=False)

    Aeig_CHIEF = torch.linalg.eigvals(A_CHIEF)
    Aeig_abs_CHIEF = Aeig_CHIEF.abs()
    Aeig_abs_sorted_CHIEF ,Aeig_idx_CHIEF = torch.sort(Aeig_abs_CHIEF)
    Aeig_sorted_CHIEF = Aeig_CHIEF[:,Aeig_idx_CHIEF.squeeze()]
    Aeigs_CHIEF_abs.append(Aeig_abs_sorted_CHIEF[:,-1])

    Heig_CHIEF = H_CHIEF * Aeig_sorted_CHIEF[:, -1].item()

    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)
    Eeig_CHIEF = compute_E(reflector, p, board, H=Heig_CHIEF)


    pressure = torch.abs(E@x).mean()
    pressure_CHIEF = torch.abs(E_CHIEF@x).mean()
    
    pressure_eig = torch.abs(Eeig@x).mean()
    pressure_eig_CHIEF = torch.abs(Eeig_CHIEF@x).mean()

    ps.append(pressure.item())
    ps_CHIEF.append(pressure_CHIEF.item())    
    
    ps_eig.append(pressure_eig.item())
    ps_eig_CHIEF.append(pressure_eig_CHIEF.item())
    
    Aeigs.append(Aeig_sorted[:,-1].imag.item())
    Aeigs_CHIEF.append(Aeig_sorted_CHIEF[:,-1].imag.item())
    
    Aeigs_real.append(Aeig_sorted[:,-1].real.item())
    Aeigs_CHIEF_real.append(Aeig_sorted_CHIEF[:,-1].real.item())

# log_list = lambda x: [math.log(i) for i in x]


# print(Aeigs)
list_real = lambda x: [i.real for i in x[0]]
list_imag = lambda x: [i.imag for i in x[0]]

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(ds, ps, label = f'H - A LU')
plt.plot(ds, ps_CHIEF, label = f'H - A CHIEF LU')
plt.plot(ds, ps_eig, label = f'$\lambda H$ - A LU')
plt.plot(ds, ps_eig_CHIEF, label = f'$\lambda H$ - A CHIEF LU')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Diameter (m)')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(Aeigs_abs, ps, label = f'A LU')
plt.scatter(Aeigs_CHIEF_abs, ps_CHIEF, label = f'A CHIEF LU')
plt.ylabel('Pressure (Pa)')
plt.xlabel('$max(|\lambda|)$')
plt.legend()




plt.show()