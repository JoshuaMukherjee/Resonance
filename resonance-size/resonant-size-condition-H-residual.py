from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points
from acoustools.Constants import wavelength
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A, augment_A_CHIEF
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt
import scipy
import math

board = TRANSDUCERS
p = create_points(1,1,0,0,-0.002)

path = "../BEMMedia"

x = iterative_backpropagation(p, board=board)

N = 100
ds = [0.01 + (0.02) * i/N for i in range(N)]
# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]o



Hreses = []
Hreses_LU = []
Hreses_CHIEF = []
Hreses_CHIEF_LU = []
Hreses_CHIEF_rect = []

Hsv = []
Hsv_CHIEF = []
Hsv_CHIEF_LU = []
Hsv_CHIEF_rect = []


ps = []
ps_LU = []
ps_CHIEF = []
ps_CHIEF_LU = []
ps_CHIEF_rect = []


for i,d in enumerate(ds):

    print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)
    p = get_tetra_centroids(reflector)


    A = compute_A(reflector)
    H,_,B,_ = compute_H(reflector, board, A=A, use_LU=False, use_OLS=True, return_components=True)
    residual = A@H - B
    E = compute_E(reflector, p, board, H=H)
    H_LU = compute_H(reflector, board, A=A, use_LU=True, use_OLS=False)
    residual_LU = A@H_LU - B
    E_LU = compute_E(reflector, p, board, H=H_LU)

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    A_CHIEF = augment_A_CHIEF(A, internal_points, scatterer=reflector)
    H_CHIEF, _, B_CHIEF, H_CHIEF_exp = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, use_LU=False, use_OLS=True, return_components=True)
    residual_CHIEF = A@H_CHIEF - B
    # min_sv_CHIEF = torch.linalg.svdvals(A_CHIEF).min()
    E_CHIEF = compute_E(reflector, p, board, H=H_CHIEF)

    H_CHIEF_LU, _, B_CHIEF_LU, H_CHIEF_LU_exp = compute_H(reflector, board, A=A_CHIEF, internal_points=internal_points, use_LU=True, use_OLS=False, return_components=True)
    residual_CHIEF_LU = A@H_CHIEF_LU - B
    # min_sv_CHIEF_LU = min_sv_CHIEF
    E_CHIEF_LU = compute_E(reflector, p, board, H=H_CHIEF_LU)


    A_CHIEF_rect = augment_A_CHIEF(A, internal_points, scatterer=reflector, CHIEF_mode='rect')
    H_CHIEF_rect, _, B_CHIEF_rect, H_CHIEF_rect_exp  = compute_H(reflector, board, A=A_CHIEF_rect , internal_points=internal_points, CHIEF_mode='rect', use_LU=False, use_OLS=True, return_components=True)
    residual_CHIEF_rect = A@H_CHIEF_rect - B
    # min_sv_CHIEF_rect  = torch.linalg.svdvals(A_CHIEF_rect ).min()
    E_CHIEF_rect  = compute_E(reflector, p, board, H=H_CHIEF_rect )


    pressure = torch.abs(E@x).mean()
    pressure_LU = torch.abs(E_LU@x).mean()
    pressure_CHIEF = torch.abs(E_CHIEF@x).mean()
    pressure_CHIEF_LU = torch.abs(E_CHIEF_LU@x).mean()
    pressure_CHIEF_rect = torch.abs(E_CHIEF_rect@x).mean()
    

    ps.append(pressure.item())
    ps_LU.append(pressure_LU.item())
    ps_CHIEF.append(pressure_CHIEF.item())
    ps_CHIEF_LU.append(pressure_CHIEF_LU.item())
    ps_CHIEF_rect.append(pressure_CHIEF_rect.item())

    Hreses.append(residual.abs().sum().item().real)
    Hreses_LU.append(residual_LU.abs().sum().item().real)
    Hreses_CHIEF.append(residual_CHIEF.abs().sum().item().real)
    Hreses_CHIEF_LU.append(residual_CHIEF_LU.abs().sum().item().real)
    Hreses_CHIEF_rect.append(residual_CHIEF_rect.abs().sum().item().real)


    # Hsv.append(min_sv.item())
    # Hsv_CHIEF.append(min_sv_CHIEF.item())
    # Hsv_CHIEF_LU.append(min_sv_CHIEF_LU.item())
    # Hsv_CHIEF_rect.append(min_sv_CHIEF_rect.item())

# log_list = lambda x: [math.log(i) for i in x]
# round_elem  = lambda x: float('%.2g' % x)

# _, _, Ar, Ap, _ = scipy.stats.linregress(log_list(Hreses), log_list(ps))
# _, _, Ar_LU, Ap_LU, _ = scipy.stats.linregress(log_list(Hreses_LU), log_list(ps_LU))
# _, _, Ar_CHIEF, Ap_CHIEF, _ = scipy.stats.linregress(log_list(Hreses_CHIEF), log_list(ps_CHIEF))
# _, _, Ar_CHIEF_LU, Ap_CHIEF_LU, _ = scipy.stats.linregress(log_list(Hreses_CHIEF_LU), log_list(ps_CHIEF_LU))
# _, _, Ar_CHIEF_rect, Ap_CHIEF_rect, _ = scipy.stats.linregress(log_list(Hreses_CHIEF_rect), log_list(ps_CHIEF_rect))


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
plt.scatter(Hreses, ps, label = f'A OLS')
plt.scatter(Hreses_LU, ps_LU, label = f'A LU')
plt.scatter(Hreses_CHIEF, ps_CHIEF, label = f'A CHIEF OLS')
plt.scatter(Hreses_CHIEF_LU, ps_CHIEF_LU, label = f'A CHIEF LU')
plt.scatter(Hreses_CHIEF_rect, ps_CHIEF_rect, label = f'A CHIEF rect OLS')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Residual')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.subplot(2,1,2)
# plt.scatter(Hreses, ps, label = f'A OLS, r = {round_elem(Ar)}, p = {round_elem(Ap)}')
# plt.scatter(Hreses_LU, ps_LU, label = f'A LU, r = {round_elem(Ar_LU)}, p = {round_elem(Ap_LU)}')
# plt.scatter(Hreses_CHIEF, ps_CHIEF, label = f'A CHIEF OLS, r = {round_elem(Ar_CHIEF)}, p = {round_elem(Ap_CHIEF)}')
# plt.scatter(Hreses_CHIEF_LU, ps_CHIEF_LU, label = f'A CHIEF LU, r = {round_elem(Ar_CHIEF_LU)}, p = {round_elem(Ap_CHIEF_LU,)}')
# plt.scatter(Hreses_CHIEF_rect, ps_CHIEF_rect, label = f'A CHIEF rect OLS r = {round_elem(Ar_CHIEF_rect)}, p = {round_elem(Ap_CHIEF_rect)}')
# plt.ylabel('Pressure (Pa)')
# plt.xlabel('Residual')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()

# plt.subplot(4,1,2)
# plt.scatter(ds, Hconds, label = 'A')
# plt.scatter(ds, Hconds_CHIEF, label = 'A CHIEF')
# plt.scatter(ds, Hconds_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Diameter (m)')
# plt.ylabel('Cond')
# plt.legend()

# plt.subplot(3,1,2)
# plt.scatter(Hsv, Hconds,label = 'A')
# plt.scatter(Hsv_CHIEF, Hconds_CHIEF, label = 'A CHIEF OLS')
# plt.scatter(Hsv_CHIEF_LU, Hconds_CHIEF_LU, label = 'A CHIEF LU')
# plt.scatter(Hsv_CHIEF_rect, Hconds_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Cond')
# plt.ylabel('Min singular Value')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()

# plt.subplot(3,1,3)
# plt.scatter(Hsv, ps,label = 'A')
# plt.scatter(Hsv_CHIEF, ps_CHIEF, label = 'A CHIEF OLS')
# plt.scatter(Hsv_CHIEF_LU, ps_CHIEF_LU, label = 'A CHIEF LU')
# plt.scatter(Hsv_CHIEF_rect, ps_CHIEF_rect, label = 'A CHIEF rect')
# plt.xlabel('Min singular Value')
# plt.ylabel('Pressure (Pa)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()



plt.show()