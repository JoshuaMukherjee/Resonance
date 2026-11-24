from acoustools.Mesh import load_scatterer, get_CHIEF_points, get_centres_as_points, scale_to_diameter
from acoustools.BEM import compute_H
from acoustools.Utilities import BOTTOM_BOARD
import torch, time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


board = BOTTOM_BOARD

root = '../BEMMedia/'

meshes = ['Sphere-lam1.stl', 'Octohedron_lam2.stl', 'Cube_Angle_lam2.stl']
labels = ['Sphere', 'Octahedron', 'Cube']

N = 1001
points = torch.linspace(0, 1000, steps=N).tolist()
# points = [0,] + points

TRIALS = 30

for label,mesh in zip(labels,meshes):
    times = []
    mins = []
    maxs = []
    stds = []
    scatterer = load_scatterer(root + mesh)
    scale_to_diameter(scatterer, 0.02)
    mesh_elements = get_centres_as_points(scatterer).shape[2]
    for p in points:
        ts = []
        for t in range(TRIALS):
            print(t, label, p, end='\r')
            P = int(p)
            if P == 0: 
                internal_points = None
            else:
                internal_points = get_CHIEF_points(scatterer,  P=P, method='tetra-random')
        
            start = time.time_ns()
            H = compute_H(scatterer,board=board, internal_points=internal_points)
            end = time.time_ns()
        
            ts.append((end-start))
        
        mins.append(min(ts))
        maxs.append(max(ts))
        times.append(sum(ts)/TRIALS)
        stds.append(np.std(ts))
    
    plt.plot(points, times, label=label + f' M={mesh_elements}')
    
    
    # top = np.array(maxs)
    # bottom =  np.array(mins)
    
    top = np.array(times) + np.array(stds)
    bottom = np.array(times) - np.array(stds)
    
    
    plt.fill_between(points, top, bottom, alpha = 0.3)
    
    print()
plt.legend()
plt.xlabel("Internal Points")
plt.ylabel("H Computation Time (ns)")
plt.show()
    
