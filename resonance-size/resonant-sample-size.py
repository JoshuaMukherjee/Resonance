from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt


board = TRANSDUCERS
p = create_points(1,1,0,0,-0.02)

path = "../BEMMedia"

x = iterative_backpropagation(p, board=board)

ds = [0.01+wavelength * i/40 for i in range(10, 100)]
fracs = [i/100 for i in range(2,10)]

# ds = [wavelength * i/10 for i in range(10, 50, 10)]
# fracs = [0.1, 0.5]



for frac in fracs:
    pressures = []
    for i,d in enumerate(ds):

        print(frac, i, end='\t\r')

        # d = wavelength*2
        reflector = load_scatterer(path + '/sphere-lam2.stl')
        scale_to_diameter(reflector, d)
        centre_scatterer(reflector)


        centres = get_tetra_centroids(reflector)
        N = centres.shape[2]
        sample_size = int(N * frac)
        sample = centres[:,:,torch.randint(0, N, (1,sample_size)).squeeze()]

        H = get_cache_or_compute_H(reflector, board, path=path, use_cache_H=False)

        pressure_CHIEF = propagate_BEM_pressure(x, sample, H=H, path=path, scatterer=reflector, board=board)

        pressures.append(pressure_CHIEF.mean().item())


    plt.plot(ds, pressures, label=frac)

plt.ylabel("Pressure (Pa)")
plt.xlabel("Diameter (m)")
plt.legend()
plt.show()
