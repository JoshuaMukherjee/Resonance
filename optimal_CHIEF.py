
if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device, DTYPE
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_scatterer,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points, get_centre_of_mass_as_points, get_normals_as_points, get_tetra_centroids
    from acoustools.BEM import propagate_BEM_pressure, compute_E, compute_A, augment_A_CHIEF, compute_H, propagate_BEM_laplacian_abs
    from acoustools.Constants import wavelength,k, P_ref

    import torch

    from torch import Tensor
    from vedo import Mesh



    def find_optimal_CHIEF_points(scatterer: Mesh, board:Tensor, k:float=k, p_ref = P_ref, start_p = None, start_N = 20, max_N = 10, 
                                  steps= 50, lr=1e-3, log=False, optimiser:torch.optim.Optimizer=torch.optim.NAdam, path='', frac=0.05):
        '''
        @private
        
        :param scatterer: Description
        :type scatterer: Mesh
        :param board: Description
        :type board: Tensor
        :param k: Description
        :type k: float
        :param p_ref: Description
        :param start_p: Description
        :param max_N: Description
        :param steps: Description
        :param lr: Description
        :param log: Description
        :param optimiser: Description
        :type optimiser: torch.optim.Optimizer
        '''

        # point = start_p if start_p is not None else torch.tensor(scatterer.generate_random_points(1).points).unsqueeze(2).permute(2,1,0)
        com = get_centre_of_mass_as_points(scatterer)
        # point = com.detach().clone()
        # point   = get_CHIEF_points(scatterer, P = start_N, start='centre', method='random', scale = 0.01, scale_mode='diameter-scale') 


        
        centres = get_tetra_centroids(scatterer)
        N = centres.shape[2]
        sample_size = int(N * frac)
        sample_points = centres[:,:,torch.randint(0, N, (1,sample_size)).squeeze()]

        point = centres[:,:,0:start_N]

        startA = compute_A(scatterer, k=k)


        areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
        centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
        norms = get_normals_as_points(scatterer, permute_to_points=False)

        x = iterative_backpropagation(com, board=board, k=k, p_ref=p_ref)



        for j in range(1):
            if j != 0: point = torch.cat([point, com.detach().clone() + create_points(1,1,max_pos=1e-4, min_pos=-1e-4)], dim=2).detach()
            print(point.shape)
            # if j != 0: 
                # p = torch.tensor(scatterer.generate_random_points(1).points).unsqueeze(2)
                # point = torch.cat([point, p], dim=2).detach()
            point =  point.requires_grad_() 
            optim = optimiser([point],lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=10, gamma=0.75)

            for i in range(steps):

                
                # print(point)

                optim.zero_grad()       


                A = augment_A_CHIEF(startA.clone(), internal_points=point, k=k, scatterer=scatterer, centres=centres, areas=areas, norms=norms)
                # A = compute_A(scatterer, k=k,internal_points=point, CHIEF_mode='rect')
                H = compute_H(scatterer, board, p_ref=p_ref, k=k, A=A, internal_points=point, use_LU=False, use_OLS=True)
                
                # Eint = compute_E(scatterer, point, board, use_cache_H=False, H=H, k=k, p_ref=p_ref)
                pressure = propagate_BEM_pressure(x, sample_points, scatterer, board, H=H, k=k, p_ref=p_ref)

                # objective = torch.max(pressure)
                objective = torch.mean(pressure) 
                if log: print(j+1,i, objective.item())

             
                    # print(best_pt)

                objective.backward()
                

                optim.step()

                scheduler.step()

            
        return point
    
    board = TOP_BOARD

    path = "../BEMMedia/"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    scatterer_path = 'blob.stl'
    # scatterer_path = 'Sphere-lam2.stl'
    
    scatterer = load_scatterer(path + scatterer_path)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*2
 
    # d = wavelength * 2
    # d = wavelength * 1.2345
    # d = 0.0234ß
    # d = wavelength+0.001
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)

    start_N = 3

    # internal_points  = get_CHIEF_points(scatterer, P = 20, start='centre', method='uniform')
   


    # print(internal_points.shape)
    optimal_points = find_optimal_CHIEF_points(scatterer, board, log=True, path=path, start_N=start_N)

    end_N = optimal_points.shape[2]

    print(optimal_points)


    internal_points  = get_CHIEF_points(scatterer, P = end_N , start='centre', method='random', scale = 0.1, scale_mode='diameter-scale')

    print(internal_points)

    Eint,Fint,Gint,Hint = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=internal_points)

    print(internal_points.shape,"\n", optimal_points.shape)

    # print(torch.norm(optimal_points, p=2, dim=1))

    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=optimal_points)

    x = iterative_backpropagation(p, board=board)
    x =translate_hologram(x, dz=0.001, board=board)

    Visualise(*ABC(0.01, plane='xz'), x, points=optimal_points,colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure, propagate_BEM_pressure, '-'], 
              res=(100,100), link_ax=[0,1,2], arangement=(2,2),
              colour_function_args=[{'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k}, 
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":H,"internal_points":optimal_points },
                                    {'scatterer':scatterer,'board':board,'path':path,"use_cache_H":False,"p_ref":p_ref,'k':k,"H":Hint,"internal_points":internal_points },
                                    {"ids":[1,2]}], vmax=1000)
