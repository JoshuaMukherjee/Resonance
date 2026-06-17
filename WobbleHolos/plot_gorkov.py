from acoustools.Export.Holo import load_holograms
from acoustools.Export.CSV import read_phases_from_file
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list, create_points, TOP_BOARD
from acoustools.Paths import interpolate_points,interpolate_circle
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import BEM_gorkov_analytical
import os, torch, pickle

import matplotlib.pyplot as plt

# folder = 'data\compare_reflector_wobble\Z-holos\BEM'


root = './Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'
extension = '.csv'

BEM_root = '../BEMMedia/'
reflector = load_scatterer(BEM_root + 'LargeTunnel-full-lam2.stl')
bounds = reflector.bounds()
scale_to_diameter(reflector, (bounds[1] - bounds[0])/1000)

H,H_CHIEF, internal_points = pickle.load(open('./Resonance/data/WT-lam2-objs.bin', 'rb'))


# start = create_points(1,1,0,-0.04,0.03)
# end = create_points(1,1,0,0.00,0.03)

# path = interpolate_points(start, end, n=1000)

# origin = create_points(0,0,0,0,0)
# path = interpolate_circle(origin, 0.01, n = 100)

N = 5000
start = create_points(1,1,0,-0.01,0.03)
end = create_points(1,1,0,0.01,0.03)
path = interpolate_points(start, end, n=N)

board = TOP_BOARD

Hs = [H, H_CHIEF, H_CHIEF]

for j,folder in enumerate([folder_BEM, folder_BEM, folder_CHIEF]):
    holos = []

    for i in range(N):
        print(i, end='\r')
        # x = load_holograms(root+folder+'/'+f)[0]
        try:
            x = read_phases_from_file(root+folder+'/'+str(i)+extension, invert=False, top_board=True)
        except FileNotFoundError:
            break
        # print(root+folder+'/'+str(i)+extension,x.sum())



        holos.append(x)

    print(holos[0].mean())

    prev = torch.zeros_like(holos[0])
    phase_changes = []
    phases = []

    Us = []
    for i,(x,p) in enumerate(zip(holos, path)):


        print(i, end='\r')
        H_ = Hs[j]
        U = BEM_gorkov_analytical(x, points=p, scatterer=reflector, path=BEM_root, board=board, H=H_)
        Us.append(U.abs().item())

           
        phase = x.angle().abs().mean()
        
        phase_change = (prev.angle() - x.angle()).abs().mean()
        # print(prev.angle()[:,0], x.angle()[:,0])
        phase_changes.append(phase_change.item())
        phases.append(phase.item())

        prev = x.clone()


    plt.subplot(2,1,1)
    plt.plot(Us)

    plt.subplot(2,1,2)
    plt.plot(phase_changes)


plt.show()