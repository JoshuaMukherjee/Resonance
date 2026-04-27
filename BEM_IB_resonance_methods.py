from acoustools.Solvers import iterative_backpropagation, translate_hologram
from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
from acoustools.Visualiser import Visualise,ABC
from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, merge_scatterers, get_centres_as_points, get_normals_as_points, get_CHIEF_points, get_centre_of_mass_as_points, insert_parasite
from acoustools.BEM import propagate_BEM_pressure, compute_E
from acoustools.Constants import wavelength,k, P_ref

import torch


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

p = create_points(1,1, y=0,x=0,z=0)

x = iterative_backpropagation(p, board=board)
x =translate_hologram(x, dz=0.001, board=board)


H_method = 'OLS'
E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True)


a = get_centre_of_mass_as_points(scatterer)
c=-1j
Eac,Fac,Gac,Hac = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, a=a,c=c)

infected_scatterer = insert_parasite(scatterer, parasite_size=d*0.7, parasite_path=scat_path)
outer_alphas = torch.ones((1,M))
parasite_alphas = torch.zeros((1,M))
infected_alphas = torch.cat((outer_alphas, parasite_alphas), dim=1)
Epar,Fpar,Gpar,Hpar = compute_E(infected_scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, alphas=infected_alphas)


internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')
Echief,Fchief,Gchief,Hchief = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, internal_points=internal_points)


Ebm,Fbm,Gbm,Hbm = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, h=1e-3, BM_alpha=(1j)/(20*k))


inner = load_multiple_scatterers(paths)
inner.flip_normals()
centre_scatterer(inner)
# print(scatterer.bounds())
inner_d = d *0.75
scale_to_diameter(inner,inner_d)
shell_scatterer = merge_scatterers(scatterer, inner)
alphas_out = torch.ones((1,M))
alphas_in = torch.zeros((1,M)) + 0.1
shell_alphas = torch.cat((alphas_out, alphas_in), dim=1)
Eshell,Fshell,Gshell,Hshell = compute_E(shell_scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method=H_method, return_components=True, alphas=shell_alphas)



Visualise(*ABC(0.03), x,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure], res=(100,100),
            colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hac, 'a':a,'c':c},
                                  {'scatterer':infected_scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hpar, 'alphas':infected_alphas},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hchief, 'internal_points':internal_points},
                                  {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hbm},
                                  {'scatterer':shell_scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hshell, "alphas":shell_alphas},
                                ], 
            titles=["BEM", 'Modified Greens Function', 'Parasitic Body', 'CHIEF', "Burton-Miller F.D.", "Shell"],
            arrangement=(2,3),
            
            vmax=500)
