from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points, get_tetra_centroids
from acoustools.Constants import wavelength
from acoustools.BEM import compute_A, augment_A_CHIEF, compute_H, propagate_BEM_pressure
from acoustools.Utilities import TRANSDUCERS, create_points
from acoustools.Solvers import kd_solver

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


board = TRANSDUCERS
x = kd_solver(create_points(1,1,0,0,0), board=board)

ds_frac = [1, 1.5, 2, 2.5 ,3 ]
ds = [d * wavelength for d in ds_frac]

# points = [1, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000]
points = [1, 5]


repeats = 2

path = '../BEMMedia/'
scatterer_name = 'sphere-lam4.stl'


for i,d in enumerate(ds):
    pressures = []
    pressure_to_plot = []
    for rep in range(repeats):
        pressures.append([])

        scatterer = load_scatterer(path+scatterer_name, root_path=path)
        centre_scatterer(scatterer)
        scale_to_diameter(scatterer, d)

        A = compute_A(scatterer)
        p = get_tetra_centroids(scatterer)

       

        for P in points:
            print(d, P, end='\r')

            internal_points = get_CHIEF_points(scatterer, P=P, method = 'tetra-random')
            A_CHIEF = augment_A_CHIEF(A, internal_points=internal_points, scatterer=scatterer)
            H = compute_H(scatterer, board=board, A=A_CHIEF, internal_points=internal_points)
            press = propagate_BEM_pressure(x, points=p, board=board, H=H, scatterer=scatterer, path=path).mean()
            pressures[i].append(press.item())
        pressure_to_plot.append(sum(pressures[i]) / repeats)
    plt.plot(points, pressures, marker='x', label=f'${ds_frac[i]}\lambda$')


plt.xlabel("Number of Internal Points")
plt.ylabel("Mean Internal Pressure (Pa)")
plt.legend(title='Sphere Diameter')
plt.show()
