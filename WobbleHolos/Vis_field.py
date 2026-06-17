from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list, create_points, TOP_BOARD
from acoustools.Paths import interpolate_points
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import BEM_gorkov_analytical
import os, torch, pickle

import matplotlib.pyplot as plt

# folder = 'data\compare_reflector_wobble\Z-holos\BEM'


root = './Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'

BEM_root = '../BEMMedia/'
reflector = load_scatterer(BEM_root + 'LargeTunnel-full-lam2.stl')
bounds = reflector.bounds()
scale_to_diameter(reflector, (bounds[1] - bounds[0])/1000)


H,H_CHIEF, internal_points = pickle.load(open('./Resonance/data/WT-lam2-objs.bin', 'rb'))


start = create_points(1,1,0,-0.04,0.03)
end = create_points(1,1,0,0.00,0.03)

path = interpolate_points(start, end, n=1000)

board = TOP_BOARD


I = 75

for folder in [folder_BEM, folder_CHIEF]:
    holos = []

    for i,f in enumerate(os.listdir(root+folder)):
        print(i, end='\r')
        if f[0] != '.':
            x = load_holograms(root+folder+'/'+f)[0]
            holos.append(x)

    print(holos[0].mean())

    Us = []
    for i,(x,p) in enumerate(zip(holos, path)):
    
        if i == I:
            plt.gcf().clear()
            Visualise(*ABC(0.04, origin=p, plane='xy'), x, colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical], 
                      colour_function_args=[{'path':BEM_root,'scatterer':reflector, 'board':board,'H':H_CHIEF}, 
                                            {'path':BEM_root,'scatterer':reflector, 'board':board,'H':H}],
                    cmaps=['seismic', 'seismic'],
                                            
                                            )


    plt.plot(Us)


plt.show()