from acoustools.Utilities import create_points, TRANSDUCERS, device
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_tetra_centroids, get_CHIEF_points 
from acoustools.Constants import wavelength
from acoustools.BEM import get_cache_or_compute_H, compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation

import torch
import matplotlib.pyplot as plt

torch.set_printoptions(precision=7)

board = TRANSDUCERS


path = "../BEMMedia"

reflector = load_scatterer(path + '/sphere-lam2.stl')

d = wavelength*2

scale_to_diameter(reflector, d)
centre_scatterer(reflector)

centres = get_tetra_centroids(reflector)


positions = []

optimiser = torch.optim.Adam
lr = 1e-3
epochs = 100

param = centres[:,:,0:2].clone() 
param.requires_grad_()


target = create_points(2,1,0,0,[0.00,-0.002])

optim = optimiser([param],lr)
# exit()

positions.append(param.clone().cpu().detach().squeeze())

def objective(point, target):
    return torch.norm(point-target, p=2).squeeze()

def snap_to_grid(point, centres):
    vec = (point.unsqueeze(3) - centres.unsqueeze(2))
    distances = torch.norm(vec, p=2, dim=1)
    min_distance = distances.min(dim=1).values

    
    return min_distance.mean()

alpha = 1e0

for i in range(epochs):
    
    optim.zero_grad()

    obj = objective(param,target=target) 
    snap = snap_to_grid(param, centres)
    # if obj < 1e-4:
    obj = obj + alpha * snap

    obj.backward()
    optim.step()

    positions.append(param.clone().cpu().detach().squeeze())




fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs1 = []
ys1 = []
zs1 = []

xs2 = []
ys2 = []
zs2 = []

for p in positions:
    xs1.append( p[0,0].cpu().detach().real)
    ys1.append( p[1,0].cpu().detach().real)
    zs1.append( p[2,0].cpu().detach().real)

    xs2.append( p[0,1].cpu().detach().real)
    ys2.append( p[1,1].cpu().detach().real)
    zs2.append( p[2,1].cpu().detach().real)


ax.scatter(centres[0,0,:].cpu().detach().real, centres[0,1,:].cpu().detach().real, centres[0,2,:].cpu().detach().real, alpha=0.01)

ax.plot(xs1,ys1,zs1)
ax.plot(xs2,ys2,zs2)

target = target.squeeze().cpu()
ax.plot(target[0,0].item(),target[1,0].item(),target[2,0].item(), marker='x', color='red')
ax.plot(target[0,1].item(),target[1,1].item(),target[2,1].item(), marker='x', color='red')
ax.scatter(xs1[-1],ys1[-1],zs1[-1], color='purple', marker='x')
ax.scatter(xs2[-1],ys2[-1],zs2[-1], color='purple', marker='x')
plt.show()
