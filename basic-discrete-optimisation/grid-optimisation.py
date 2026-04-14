from acoustools.Utilities import create_points, TRANSDUCERS, device
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt

torch.set_printoptions(precision=7)

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
lr = 1e-3
epochs = 1000

param = centres[:,:,0].clone() + 0.03
param.requires_grad_()
print(param)

target = create_points(1,1, 0.001, 0,0)

optim = optimiser([param],lr)


positions.append(param.clone().cpu().detach().squeeze())

def objective(point, target):
    return torch.norm(point.unsqueeze(2)-target, p=2).squeeze()

def snap_to_grid(point, centres):
    vec = (point.unsqueeze(2) - centres)
    distances = torch.norm(vec, p=2, dim=1)
    min_distance = distances.min()
    
    print(min_distance)
    
    return min_distance

alpha = 1e1

for i in range(epochs):
    
    optim.zero_grad()

    obj = objective(param,target=target) 
    snap = snap_to_grid(param, centres)
    # if obj < 1e-4:
    obj = obj + alpha * snap

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


ax.scatter(centres[0,0,:].cpu().detach().real, centres[0,1,:].cpu().detach().real, centres[0,2,:].cpu().detach().real, alpha=0.01)

ax.plot(xs,ys,zs)
target = target.squeeze().cpu()
ax.plot(target[0].item(),target[1].item(),target[2].item(), marker='x', color='red')
ax.scatter(xs[-1],ys[-1],zs[-1], color='purple', marker='x')
plt.show()
