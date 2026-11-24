from acoustools.Mesh import load_scatterer, get_CHIEF_points, get_centres_as_points, scale_to_diameter
from acoustools.BEM import compute_H
from acoustools.Utilities import BOTTOM_BOARD
import torch, time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


board = BOTTOM_BOARD

root = '../BEMMedia/'

meshes = ['Sphere-lam1.stl','Sphere-lam2.stl','Sphere-lam4.stl', 'Octohedron_lam2.stl', 'Cube_Angle_lam2.stl']
labels = ['Sphere', 'Sphere', 'Sphere', 'Octahedron', 'Cube']

N = 100
points = torch.linspace(0, 1000, steps=N).tolist()
# points = [0,] + points

TRIALS = 5

for label,mesh in zip(labels,meshes):
    
    ranks = []
    mins = []
    maxs = []
    stds = []
    scatterer = load_scatterer(root + mesh)
    scale_to_diameter(scatterer, 0.02)
    mesh_elements = get_centres_as_points(scatterer).shape[2]
    point_as_rank = [p/ mesh_elements for p in points]
    for p in points:
        rs = []
        for t in range(TRIALS):
            print(t, label, p, end='\r')
            P = int(p)
            if P == 0: 
                internal_points = None
            else:
                internal_points = get_CHIEF_points(scatterer,  P=P, method='tetra-random')
            
        
            start = time.time_ns()
            H, A,_,_ = compute_H(scatterer,board=board, internal_points=internal_points, return_components=True)
            end = time.time_ns()

            rs.append((torch.linalg.matrix_rank(A) / A.shape[2]).item())
            
            
        
        mins.append(min(rs))
        maxs.append(max(rs))
        ranks.append(sum(rs)/TRIALS)
        stds.append(np.std(rs))
    
    plt.plot(points, ranks, label=label + f' M={mesh_elements}')
    
    
    # top = np.array(maxs)
    # bottom =  np.array(mins)
    
    top = np.array(ranks) + np.array(stds)
    bottom = np.array(ranks) - np.array(stds)
    
    
    plt.fill_between(points, top, bottom, alpha = 0.3)
    
    print()
plt.legend()
plt.xlabel("Internal Points")
plt.ylabel("Rank(A) / M")
plt.show()
    
