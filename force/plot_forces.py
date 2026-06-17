import matplotlib.pyplot as plt
import pandas as pd
from acoustools.Constants import wavelength
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})

N= 250

data = pd.read_csv('Resonance/data/outputs/force/ARF.csv')

radius = [d /( 2*wavelength) for d in data['Diameter (m) ']]
force = data['ARF CHIEF (P=0) (N)']




sh_data = pd.read_csv('Resonance/data/outputs/force/SH-Force.csv')
sh_radius = [d /( wavelength) for d in sh_data['Diameter (lam)']]
sh_force = sh_data['SH Force (N)']


data = pd.read_csv('Resonance/data/outputs/force/ARF-CHIEF.csv')

radiusCHIEF = [d /( 2*wavelength) for d in data['Diameter (m) ']]
forceCHIEF = data['ARF CHIEF (P=50) (N)']
U = data[' -grad U (N)']

plt.plot(radius[:N], U[:N], label='-$\\nabla$U', linestyle=':')
plt.plot(sh_radius[:N], sh_force[:N],label='Spherical Harmonics')
plt.plot(radius[:N], force[:N], label='BEM')
plt.plot(radiusCHIEF[:N], forceCHIEF[:N], label='BEM-CHIEF (This Work)')


 
plt.xlabel('Radius ($\lambda$)')
plt.ylabel("Acoustic Force (N)")

plt.legend()

plt.ylim(-2.5e-3, 2.5e-3)
plt.show()