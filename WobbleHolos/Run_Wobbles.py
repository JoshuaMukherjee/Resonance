from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list
import os, time

# folder = 'data\compare_reflector_wobble\Z-holos-phase\CHIEF'
folder = 'data/compare_reflector_wobble/Z/Z-holos-phase\CHIEF'

holos = []

for i,f in enumerate(os.listdir(folder)):
    print(i, end='\r')
    x = load_holograms(folder+'/'+f)[0]
    holos.append(x)
# holos.reverse()
# exit()
print(len(holos))
with LevitatorController(ids=(999)) as lev:

    lev.set_frame_rate(200)

    lev.levitate(holos[0])

    input()
    for holo in batch_list(holos[0:70], 1):
        if len(holo) > 0:
            lev.levitate(holo)
        input()
            # time.sleep(0.1)
    print('Done')
    input()
