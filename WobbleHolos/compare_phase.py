from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import BEM_gorkov_analytical
import os, torch

import matplotlib.pyplot as plt

# folder = 'data\compare_reflector_wobble\Z-holos\BEM'


root = './Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'


for folder in [folder_BEM, folder_CHIEF]:
    holos = []

    for i,f in enumerate(os.listdir(root+folder)):
        print(i, end='\r')
        if f[0] != '.':
            x = load_holograms(root+folder+'/'+f)[0]
            holos.append(x)

    print(holos[0].mean())

    prev = torch.zeros_like(holos[0])
    phase_changes = []
    phases = []
    for i,x in enumerate(holos):
        
        phase = x.angle().abs().mean()
        
        phase_change = (prev.angle() - x.angle()).abs().mean()
        phase_changes.append(phase_change.item())
        phases.append(phase.item())

        
        print(i,phase_change)


        prev = x.clone()



    plt.subplot(2,1,1)
    plt.plot(phase_changes, label=folder)
    plt.subplot(2,1,2)
    plt.plot(phases, label=folder)

plt.subplot(2,1,1)
plt.xlabel('Frame')
plt.ylabel('Mean phase change (rad)')
plt.legend()

plt.subplot(2,1,2)
plt.xlabel('Frame')
plt.ylabel('Mean phase (rad)')
plt.legend()

plt.show()