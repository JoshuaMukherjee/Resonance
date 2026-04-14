from acoustools.Utilities import create_points, TRANSDUCERS, device
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt

board = TRANSDUCERS
p = create_points(1,1,0,0,-0.02)

path = "../BEMMedia"

reflector = load_scatterer(path + '/sphere-lam2.stl')

d = wavelength*2

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

centres = get_tetra_centroids(reflector)
print(centres)


positions = []

optimiser = torch.optim.Adam
lr = 1e-4
epochs = 100

param = centres[:,:,0].clone().requires_grad_()
optim = optimiser([param],lr)

def objective(point):
    return torch.norm(point, p=2).squeeze()

for i in range(epochs):
    
    optim.zero_grad()

    obj = objective(param)
    print(obj)

    obj.backward()
    optim.step()

    positions.append(param.clone().cpu().detach().squeeze())

print(param)  



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = []
ys = []
zs = []
for p in positions:
    xs.append( p[0].cpu().detach().real)
    ys.append( p[1].cpu().detach().real)
    zs.append( p[2].cpu().detach().real)

ax.plot(xs,ys,zs, marker='x')
ax.plot(0,0,0, marker='x', color='red')
plt.show()
