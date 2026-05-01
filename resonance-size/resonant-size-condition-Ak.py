from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points
from acoustools.Constants import wavelength, k
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A, augment_A_CHIEF, propagate_BEM_laplacian_abs
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt
import scipy
import math

board = TRANSDUCERS
p = create_points(1,1,0,0,-0.002)

path = "../BEMMedia"

x = iterative_backpropagation(p, board=board)

N = 300
ds = [0.01 + (0.02) * i/N for i in range(N)]
# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]o


Aconds = []
Aconds_CHIEF = []
Aconds_CHIEF_LU = []
Aconds_CHIEF_rect = []

Aks = []
Aks_LU = []
Aks_CHIEF = []
Aks_CHIEF_LU = []
Aks_CHIEF_rect = []


ps = []
ps_LU = []
ps_CHIEF = []
ps_CHIEF_LU = []
ps_CHIEF_rect = []

M = 400

for i,d in enumerate(ds):

    print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)
    p = get_tetra_centroids(reflector)[:,:,:M]


    A = compute_A(reflector)
    H = compute_H(reflector, board, A=A, use_LU=False, use_OLS=True)
    E = compute_E(reflector, p, board, H=H)
    H_LU = compute_H(reflector, board, A=A, use_LU=True, use_OLS=False)
    E_LU = compute_E(reflector, p, board, H=H_LU)
    

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    H_CHIEF = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, use_LU=False, use_OLS=True)
    # min_sv_CHIEF = torch.linalg.svdvals(A_CHIEF).min()
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    H_CHIEF_LU = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, use_LU=True, use_OLS=False)
    # min_sv_CHIEF_LU = min_sv_CHIEF
    E_CHIEF_LU = compute_E(reflector, p, board, H=H_CHIEF_LU)


    A_CHIEF_rect = augment_A_CHIEF(A, internal_points, scatterer=reflector, CHIEF_mode='rect')
    H_CHIEF_rect  = compute_H(reflector, board, A=A_CHIEF_rect , internal_points=internal_points, CHIEF_mode='rect', use_LU=False, use_OLS=True)
    # min_sv_CHIEF_rect  = torch.linalg.svdvals(A_CHIEF_rect ).min()
    E_CHIEF_rect  = compute_E(reflector, p, board, H=H_CHIEF_rect )


    pressure = torch.abs(E@x).mean()
    pressure_LU = torch.abs(E_LU@x).mean()
    pressure_CHIEF = torch.abs(E_CHIEF@x).mean()
    pressure_CHIEF_LU = torch.abs(E_CHIEF_LU@x).mean()
    pressure_CHIEF_rect = torch.abs(E_CHIEF_rect@x).mean()

    
    Alap = propagate_BEM_laplacian_abs(x, p, reflector, board, H=H, path=path)
    Alap_LU = propagate_BEM_laplacian_abs(x, p, reflector, board, H=H_LU, path=path)
    Alap_CHIEF = propagate_BEM_laplacian_abs(x, p, reflector, board, H=H_CHIEF, path=path)
    Alap_CHIEF_LU = propagate_BEM_laplacian_abs(x, p, reflector, board, H=H_CHIEF_LU, path=path)
    Alap_CHIEF_rect = propagate_BEM_laplacian_abs(x, p, reflector, board, H=H_CHIEF_rect, path=path)
    
    Ak = torch.sqrt(Alap / pressure).mean() / k
    Ak_LU = torch.sqrt(Alap_LU / pressure_LU).mean() / k
    Ak_CHIEF = torch.sqrt(Alap_CHIEF / pressure_CHIEF).mean() / k
    Ak_CHIEF_LU = torch.sqrt(Alap_CHIEF_LU / pressure_CHIEF_LU).mean() / k
    Ak_CHIEF_rect = torch.sqrt(Alap_CHIEF_rect / pressure_CHIEF_rect).mean() / k

    del Alap,  Alap_LU, Alap_CHIEF, Alap_CHIEF_LU, Alap_CHIEF_rect

    Aks.append(Ak.item())
    Aks_LU.append(Ak_LU.item())
    Aks_CHIEF.append(Ak_LU.item())
    Aks_CHIEF_LU.append(Ak_CHIEF_LU.item())
    Aks_CHIEF_rect.append(Ak_CHIEF_rect.item())
    
    ps.append(pressure.item())
    ps_LU.append(pressure_LU.item())
    ps_CHIEF.append(pressure_CHIEF.item())
    ps_CHIEF_LU.append(pressure_CHIEF_LU.item())
    ps_CHIEF_rect.append(pressure_CHIEF_rect.item())

    Aconds.append(Acond.item())
    Aconds_CHIEF.append(Acond_CHIEF.item())
    Aconds_CHIEF_LU.append(Acond_CHIEF_LU.item())
    Aconds_CHIEF_rect.append(Acond_CHIEF_rect.item())

    

    # Asv.append(min_sv.item())
    # Asv_CHIEF.append(min_sv_CHIEF.item())
    # Asv_CHIEF_LU.append(min_sv_CHIEF_LU.item())
    # Asv_CHIEF_rect.append(min_sv_CHIEF_rect.item())

log_list = lambda x: [math.log(i) for i in x]
round_elem  = lambda x: float('%.2g' % x)

_, _, Ar, Ap, _ = scipy.stats.linregress(log_list(Aks), log_list(ps))
_, _, Ar_LU, Ap_LU, _ = scipy.stats.linregress(log_list(Aks_LU), log_list(ps_LU))
_, _, Ar_CHIEF, Ap_CHIEF, _ = scipy.stats.linregress(log_list(Aks_CHIEF), log_list(ps_CHIEF))
_, _, Ar_CHIEF_LU, Ap_CHIEF_LU, _ = scipy.stats.linregress(log_list(Aks_CHIEF_LU), log_list(ps_CHIEF_LU))
_, _, Ar_CHIEF_rect, Ap_CHIEF_rect, _ = scipy.stats.linregress(log_list(Aks_CHIEF_rect), log_list(ps_CHIEF_rect))


import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(ds, ps, label = f'A OLS')
plt.plot(ds, ps_LU, label = f'A LU')
plt.plot(ds, ps_CHIEF, label = f'A CHIEF OLS')
plt.plot(ds, ps_CHIEF_LU, label = f'A CHIEF LU')
plt.plot(ds, ps_CHIEF_rect, label = f'A CHIEF rect OLS')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Diameter (m)')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()

plt.subplot(2,1,2)
plt.scatter(Aks, ps, label = f'A OLS, r = {round_elem(Ar)}, p = {round_elem(Ap)}')
plt.scatter(Aks_LU, ps_LU, label = f'A LU, r = {round_elem(Ar_LU)}, p = {round_elem(Ap_LU)}')
plt.scatter(Aks_CHIEF, ps_CHIEF, label = f'A CHIEF OLS, r = {round_elem(Ar_CHIEF)}, p = {round_elem(Ap_CHIEF)}')
plt.scatter(Aks_CHIEF_LU, ps_CHIEF_LU, label = f'A CHIEF LU, r = {round_elem(Ar_CHIEF_LU)}, p = {round_elem(Ap_CHIEF_LU,)}')
plt.scatter(Aks_CHIEF_rect, ps_CHIEF_rect, label = f'A CHIEF rect OLS r = {round_elem(Ar_CHIEF_rect)}, p = {round_elem(Ap_CHIEF_rect)}')
plt.ylabel('Pressure (Pa)')
plt.xlabel('$k_{est} / k$')
plt.xscale('log')
plt.yscale('log')
plt.legend()

# plt.subplot(4,1,2)
# plt.scatter(ds, Aconds, label = 'A')
# plt.scatter(ds, Aconds_CHIEF, label = 'A CHIEF')
# plt.scatter(ds, Aconds_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Diameter (m)')
# plt.ylabel('Cond')
# plt.legend()

# plt.subplot(3,1,2)
# plt.scatter(Asv, Aconds,label = 'A')
# plt.scatter(Asv_CHIEF, Aconds_CHIEF, label = 'A CHIEF OLS')
# plt.scatter(Asv_CHIEF_LU, Aconds_CHIEF_LU, label = 'A CHIEF LU')
# plt.scatter(Asv_CHIEF_rect, Aconds_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Cond')
# plt.ylabel('Min singular Value')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()

# plt.subplot(3,1,3)
# plt.scatter(Asv, ps,label = 'A')
# plt.scatter(Asv_CHIEF, ps_CHIEF, label = 'A CHIEF OLS')
# plt.scatter(Asv_CHIEF_LU, ps_CHIEF_LU, label = 'A CHIEF LU')
# plt.scatter(Asv_CHIEF_rect, ps_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Min singular Value')
# plt.ylabel('Pressure (Pa)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()



plt.show()