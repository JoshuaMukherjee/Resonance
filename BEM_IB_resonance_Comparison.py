from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, merge_scatterers, get_centres_as_points, get_normals_as_points, get_CHIEF_points, get_centre_of_mass_as_points, insert_parasite
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_gorkov_analytical, propagate_BEM_phase
from acoustools.Constants import wavelength,k, P_ref

import torch

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.family' : 'times',})

board = TOP_BOARD

path = "../BEMMedia"
# paths = [path+"/Sphere-lam2.stl"]   
# scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

p_ref = 12 * 0.22

scat_path = "/Sphere-lam2.stl"
paths = [path+scat_path]

scatterer = load_multiple_scatterers(paths)
centre_scatterer(scatterer)
d = wavelength * 2
scale_to_diameter(scatterer,d)
get_edge_data(scatterer)

centres = get_centres_as_points(scatterer)
M = centres.shape[2]

p = create_points(1,1, y=0,x=0,z=-0.02)

x = iterative_backpropagation(p, board=board)
x =translate_hologram(x, dz=0.001, board=board)


H_method = 'OLS'
E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True)


# internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
internal_points = get_CHIEF_points(scatterer, P=50, method='tetra-random')
Echief,Fchief,Gchief,Hchief = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, internal_points=internal_points)



Visualise(*ABC(0.04), x,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, '-',propagate_BEM_phase,propagate_BEM_phase,'-', BEM_gorkov_analytical, BEM_gorkov_analytical, '-'], res=(100,100),
            colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hchief, 'internal_points':internal_points},
                                  {},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hchief, 'internal_points':internal_points},
                                  {'ids':[3,4]},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hchief, 'internal_points':internal_points},
                                  {'ids':[6,7]}
                                ], 
            titles=["BEM", "CHIEF", "Difference", "", "", "", "", "", ""],
            cmaps=['hot','hot','hot','hsv','hsv','hsv', 'seismic','seismic','seismic'],
            clr_labels=['Pressure (Pa)', 'Pressure (Pa)', 'Pressure Difference (Pa)', 'Phase (rad)', 'Phase (rad)', 'Phase Difference (rad)', 'Gor\'kov Potential (J)', 'Gor\'kov Potential (J)', 'Gor\'kov Potential\nDifference (J)'],
            arrangement=(3,3),
            link_ax=None,
            vmax=[1500, 1500, 400,3.1415, 3.1415, 1, 0, 0, 0],
            vmin=[0,0,-400, -3.1415,-3.1415, -1, -5e-8, -5e-8, -1e-9])
