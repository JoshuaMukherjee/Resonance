from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list
import os

folder = 'data\compare_reflector_wobble\Z-holos\BEM'

holos = []

for i,f in enumerate(os.listdir(folder)):
    print(i, end='\r')
    x = load_holograms(folder+'/'+f)[0]
    holos.append(x)

holos.reverse()
print(len(holos))
with LevitatorController(ids=(87)) as lev:

    lev.set_frame_rate(275)

    lev.levitate(holos[500])

    input()
    exit()
    for batch in batch_list(holos, batch=1):
        lev.levitate(batch)
        # input()
    print('Done')
    input()
