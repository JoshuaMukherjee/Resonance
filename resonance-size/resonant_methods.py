from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points, get_centre_of_mass_as_points, insert_parasite, get_centres_as_points, merge_scatterers
from acoustools.Constants import wavelength, k
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure, compute_H
from acoustools.Solvers import iterative_backpropagation
from acoustools.Constants import wavelength

import torch, pickle
import matplotlib.pyplot as plt


board = TRANSDUCERS
p = create_points(1,1,0,0,0)

path = "../BEMMedia"

x_focus = iterative_backpropagation(p, board=board)

N = 200
ds = [wavelength/2 + 0.03 * (i / N) for i in range(1,N)]

# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]

frac = 0.05

pressures = []
pressures_CHIEF = []
pressures_CHIEF_rect = []
pressures_ac = []
pressures_par = []
pressures_bm = []
pressures_bm_good = []
pressures_ring = []


for i,d in enumerate(ds):
    print(i, end='\r')
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)

    p = get_tetra_centroids(reflector)
    centres = get_centres_as_points(reflector)
    M = centres.shape[2]

    H = compute_H(reflector, board, use_LU=True)

    internal_points = get_CHIEF_points(reflector, P=50, start='centre', scale_mode='diameter-scale', scale=0.1)
    H_CHIEF = compute_H(reflector, board, internal_points=internal_points, use_LU=True)
    H_CHIEF_rect = compute_H(reflector, board, internal_points=internal_points, use_LU=False, use_OLS=True, CHIEF_mode='rect')



    a = get_centre_of_mass_as_points(reflector)
    c=-1j
    Hac = compute_H(reflector, board, use_LU=True, a=a, c=c)

    infected_scatterer = insert_parasite(reflector, parasite_size=d*0.7, parasite_path= '/sphere-lam2.stl')
    outer_alphas = torch.ones((1,M))
    parasite_alphas = torch.zeros((1,M))
    infected_alphas = torch.cat((outer_alphas, parasite_alphas), dim=1)
    Hpar = compute_H(infected_scatterer, board, use_LU=True, alphas = infected_alphas )

    Hbm = compute_H(reflector, board, use_LU=True, h=1e-3, BM_alpha=(1j)/(k) )
    # Hbmgood = compute_H(reflector, board, use_LU=True, h=1e-3, BM_alpha=(1j)/(20*k) )

    inner = load_scatterer(path + '/sphere-lam2.stl')
    inner.flip_normals()
    centre_scatterer(inner)
    # print(scatterer.bounds())
    inner_d = d *0.75
    scale_to_diameter(inner,inner_d)
    shell_scatterer = merge_scatterers(reflector, inner)
    alphas_out = torch.ones((1,M))
    alphas_in = torch.zeros((1,M)) + 0.1
    shell_alphas = torch.cat((alphas_out, alphas_in), dim=1)
    Hring = compute_H(shell_scatterer, board, use_LU=True, alphas=shell_alphas )

    pressure = propagate_BEM_pressure(x_focus, p, reflector, H=H, path=path , board=board).mean().item()
    pressure_CHIEF = propagate_BEM_pressure(x_focus, p, reflector, H=H_CHIEF, path=path ,board=board).mean().item()
    pressure_CHIEF_rect = propagate_BEM_pressure(x_focus, p, reflector, H=H_CHIEF_rect, path=path ,board=board).mean().item()
    pressure_ac = propagate_BEM_pressure(x_focus, p, reflector, H=Hac, path=path, board=board ).mean().item()
    pressure_par = propagate_BEM_pressure(x_focus, p, infected_scatterer, H=Hpar, path=path , board=board).mean().item()
    pressure_bm = propagate_BEM_pressure(x_focus, p, reflector, H=Hbm, path=path , board=board).mean().item()
    # pressure_bm_good = propagate_BEM_pressure(x_focus, p, reflector, H=Hbmgood, path=path , board=board).mean().item()
    pressure_ring = propagate_BEM_pressure(x_focus, p, shell_scatterer, H=Hring, path=path, board=board ).mean().item()

    pressures.append(pressure)
    pressures_CHIEF.append(pressure_CHIEF)
    pressures_CHIEF_rect.append(pressure_CHIEF_rect)
    pressures_ac.append(pressure_ac)
    pressures_par.append(pressure_par)
    pressures_bm.append(pressure_bm)
    # pressures_bm_good.append(pressure_bm_good)
    pressures_ring.append(pressure_ring)


ds = [d / wavelength for d in ds]

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


plt.plot(ds, pressures, label='BEM')
plt.plot(ds, pressures_CHIEF, label = 'CHIEF')
plt.plot(ds, pressures_CHIEF_rect, label = 'CHIEF (Rect, OLS)')
plt.plot(ds, pressures_ac, label='Modified Green')
plt.plot(ds, pressures_par, label='Parasite')
plt.plot(ds, pressures_bm, label='Burton-Miller (Finite Differences, i/k)')
# plt.plot(ds, pressures_bm_good, label='Burton-Miller (Finite Differences, i/20k)')
plt.plot(ds, pressures_ring, label='ICA-Ring')

plt.xlabel("Diameter ($\lambda$)")
plt.ylabel("Mean Internal Pressure (Pa)")

pickle.dump([pressures, pressures_CHIEF, pressures_ac, pressures_par, pressures_bm, pressures_bm_good, pressures_ring, ds], open('Pressure_vals.obj', 'wb'))

plt.legend()
plt.show()