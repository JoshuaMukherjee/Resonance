from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import compute_H, compute_E, propagate_BEM_pressure, compute_A
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



Aconds = []
Hconds = []
Econds = []

ps = []


for i,d in enumerate(ds):

    # print(d, i, end='\t\r')

    # d = wavelength*2
    reflector = load_scatterer(path + '/sphere-lam2.stl')
    scale_to_diameter(reflector, d)
    centre_scatterer(reflector)

    A = compute_A(reflector)
    Acond = torch.linalg.cond(A)

    H = compute_H(reflector, board, A=A)
    Hcond = torch.linalg.cond(H)

    E = compute_E(reflector, p, board, H=H)
    Econd = torch.linalg.cond(E)
    print(Econd)

    pressure = torch.abs(E@x)

    Aconds.append(Acond.item())
    Hconds.append(Hcond.item())
    Econds.append(Econd.item())
    ps.append(pressure.item())

Acond_max = max(Aconds)
Aconds = [Acond / Acond_max for Acond in Aconds]

Hcond_max = max(Hconds)
Hconds = [Hcond / Hcond_max for Hcond in Hconds]

Econd_max = max(Econds)
Econds = [Econd / Econd_max for Econd in Econds]


import matplotlib.pyplot as plt
plt.scatter(Aconds, ps, label = 'A')
plt.scatter(Hconds, ps, label = 'H')
plt.scatter(Econds, ps, label = 'E')
plt.legend()
plt.show()