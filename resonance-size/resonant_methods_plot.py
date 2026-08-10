import pickle

pressure, pressure_CHIEF, pressure_ac, pressure_par, pressure_bm, pressures_bm_good, pressure_ring, ds = pickle.load(open('Pressure_vals.obj', 'rb'))

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})


plt.plot(ds, pressures, label='BEM')
plt.plot(ds, pressures_CHIEF, label = 'CHIEF')
plt.plot(ds, pressures_CHIEF_rect, label = 'CHIEF (Rect, OLS)')
plt.plot(ds, pressures_ac, label='Modified Green')
plt.plot(ds, pressures_par, label='Parasite')
plt.plot(ds, pressures_bm, label='Burton-Miller (Finite Differences, i/k)')
# plt.plot(ds, pressures_bm_good, label='Burton-Miller (Finite Differences, i/20k)')
plt.plot(ds, pressures_ring, label='ICA-Ring')

plt.xlabel("Diameter ($\lambda$)")
plt.ylabel("Mean Internal Pressure (Pa)")


plt.legend()
plt.show()