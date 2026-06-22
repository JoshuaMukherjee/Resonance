from acoustools.Export.Holo import load_holograms
from acoustools.Export.CSV import read_phases_from_file
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list
import os, time

# folder = 'data\compare_reflector_wobble\Z-holos-phase\CHIEF'

# root = './Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/'
root = '.\data\compare_reflector_wobble\Z-holos-phase/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'

folder = folder_CHIEF

holos = []

N = 5000


for i in range(N):
    print(i, end='\r')
    # x = load_holograms(root+folder+'/'+f)[0]
    try:
        x = read_phases_from_file(root+folder+'/'+str(i)+'.csv', invert=False, top_board=True)
        # x = load_holograms(root+folder+'/'+str(i) + '.holo')[0]
        holos.append(x)
    except FileNotFoundError:
        break

with LevitatorController(ids=(999)) as lev:

    lev.set_frame_rate(10000)

    lev.levitate(holos[0])

    input()
    for holo in batch_list(holos[:N//2],16):
        # if len(holo) > 0
        # :
        lev.levitate(holo)
        # input()
            # time.sleep(0.1)
    print('Done')
    input()
