from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController
from acoustools.Utilities import batch_list, create_points, TOP_BOARD
from acoustools.Paths import interpolate_points
from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points
from acoustools.Visualiser import Visualise, ABC
from acoustools.BEM import BEM_gorkov_analytical, propagate_BEM_pressure
from acoustools.Export.CSV import read_phases_from_file
from acoustools.Visualiser import ABC, Visualise

import os, torch, pickle

import matplotlib.pyplot as plt

# folder = 'data\compare_reflector_wobble\Z-holos\BEM'


board = TOP_BOARD

root = './Resonance/data/compare_reflector_wobble_flat/Z/Z-holos-phase/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'

BEM_root = '../BEMMedia/'
reflector = load_scatterer(BEM_root + 'LargeTunnel-full-lam2.stl')
bounds = reflector.bounds()
scale_to_diameter(reflector, (bounds[1] - bounds[0])/1000)


data = 'data\WT-lam2-objs.bin'
H,H_CHIEF, internal_points = pickle.load(open(data, 'rb'))


root = '.\data\compare_reflector_wobble\Z-holos-phase-short/'
folder_BEM = 'BEM'
folder_CHIEF = 'CHIEF'

folder = folder_BEM


frame = 0

holos_CHIEF = read_phases_from_file(root+folder_CHIEF+'/'+str(frame)+'.csv', invert=False, top_board=True)
holos_BEM = read_phases_from_file(root+folder_BEM+'/'+str(frame)+'.csv', invert=False, top_board=True)

Visualise(*ABC(0.06, plane='xy', origin=create_points(1,1,0,0,0.03)), [holos_BEM, holos_CHIEF], res=(100,100),
          colour_functions=[BEM_gorkov_analytical, BEM_gorkov_analytical],
          colour_function_args=[{'scatterer':reflector, 'H':H_CHIEF, 'board':board, 'path':BEM_root},
                                {'scatterer':reflector, 'H':H_CHIEF, 'board':board, 'path':BEM_root}],
        cmaps=['seismic', 'seismic'],
        vmax=0,
        vmin=-1e-7
        )