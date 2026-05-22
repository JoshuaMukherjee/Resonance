from acoustools.Export.Holo import load_holograms
import os

folder = './WobbleHolos\CHIEF'

holos = []

for i,f in enumerate(os.listdir(folder)):
    print(i, end='\r')
    x = load_holograms(folder+'/'+f)[0]
    holos.append(x)

print(holos)