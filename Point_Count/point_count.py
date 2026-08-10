from acoustools.Mesh import load_scatterer, centre_scatterer, scale_to_diameter, get_CHIEF_points, get_tetra_centroids
from acoustools.Constants import wavelength
from acoustools.BEM import compute_A, augment_A_CHIEF, compute_H, propagate_BEM_pressure
from acoustools.Utilities import TRANSDUCERS, create_points
from acoustools.Solvers import kd_solver

import numpy as np
import pickle

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


board = TRANSDUCERS
x = kd_solver(create_points(1,1,0,0,0), board=board)

ds_frac = [1, 2, 3]
# ds_frac = [1, 1.5, 2 ]
ds = [d * wavelength for d in ds_frac]

# points = [1, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000]
points = [1,2,5,10,15,30,50,60,70,80,90, 100, 200, 250, 300, 350, 400]
# points = [1, 5, 10]


repeats = 50

path = '../BEMMedia/'
scatterer_name = 'sphere-lam2.stl'




for i,d in enumerate(ds):
    pressures = []
    errors = []

    scatterer = load_scatterer(path+scatterer_name, root_path=path)
    centre_scatterer(scatterer)
    scale_to_diameter(scatterer, d)

    A = compute_A(scatterer)
    p = get_tetra_centroids(scatterer)
        
    
    for rep in range(repeats):
        pressures.append([])
        errors.append([])

        for P in points:
            print(rep, d, P, end='\r')

            internal_points = get_CHIEF_points(scatterer, P=P, method = 'tetra-random')
            A_CHIEF = augment_A_CHIEF(A, internal_points=internal_points, scatterer=scatterer)
            H = compute_H(scatterer, board=board, A=A_CHIEF, internal_points=internal_points)
            press = propagate_BEM_pressure(x, points=p, board=board, H=H, scatterer=scatterer, path=path).mean()
            pressures[rep].append(press.item())
    
    pressures = np.array(pressures)

    pressures_to_plot = pressures.mean(axis=0)
    error = pressures.std(axis=0)

    print(pressures)
    print(pressures_to_plot)
    print(error)

    # plt.plot(points, pressure_to_plot, marker='x', label=f'${ds_frac[i]}\lambda$')
    y1 = pressures_to_plot + error
    y2 = pressures_to_plot - error
    
    pickle.dump([pressures, pressures_to_plot, error], open(f'{scatterer_name.replace('.','')}{ds_frac[i]}_data.pth', 'wb'))
    # exit()
    plt.plot(points, pressures_to_plot, marker='x')
    plt.fill_between(points, y1, y2 , label=f'${ds_frac[i]}\lambda$', alpha = 0.3)


plt.xlabel("Number of Internal Points")
plt.ylabel("Mean Internal Pressure (Pa)")
plt.legend(title='Sphere Diameter')
plt.show()
