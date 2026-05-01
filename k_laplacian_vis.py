if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation, translate_hologram
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise,ABC
    from acoustools.Mesh import load_multiple_scatterers,scale_to_diameter, centre_scatterer, get_edge_data, get_CHIEF_points
    from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_forward_model_second_derivative_unmixed, BEM_laplacian, propagate_BEM_laplacian_abs
    from acoustools.Constants import wavelength,k, P_ref

    import torch


    
    board = TOP_BOARD

    path = "../BEMMedia"
    # paths = [path+"/Sphere-lam2.stl"]   
    # scatterer = load_multiple_scatterers(paths,dys=[-0.06],dzs=[-0.03])

    p_ref = 12 * 0.22

    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)
    centre_scatterer(scatterer)
    print(scatterer.bounds())
    d = wavelength*2*0.71
    # d = 0.0673426732
    # d = wavelength * 2
    # d = wavelength * 1.2345
    # d = 0.01235
    # d = 0.0234ß
    # d = wavelength+0.004/
    scale_to_diameter(scatterer,d)
    get_edge_data(scatterer)


    p = create_points(1,1, y=0,x=0,z=0)

    internal_points  = get_CHIEF_points(scatterer, P = 30, start='centre', method='uniform', scale = 0.2, scale_mode='diameter-scale')

    H_method = 'OLS'
    E,F,G,H = compute_E(scatterer, p,board=board, path=path, use_cache_H=False, p_ref=p_ref,H_method='OLS', return_components=True, internal_points=internal_points)

    x = iterative_backpropagation(p, board=board, p_ref=p_ref)
    # x =translate_hologram(x, dz=0.001, board=board)

    Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(p, scatterer, board,  use_cache_H=False, H=H, p_ref=p_ref,  internal_points=internal_points)

    print(Exx@x + Eyy@x + Ezz@x)

    E_lap_sum = Exx + Eyy + Ezz

    lap = E_lap_sum @ x
    print(lap)

    E_lap = BEM_laplacian(p, scatterer, board, use_cache_H=False, H=H, p_ref=p_ref)
    print(E_lap @ x)

    print(torch.norm(E_lap,p=2))    
    
    def check_k(x):
        return torch.sqrt(x) 


    Visualise(*ABC(0.01), x, points=internal_points, res=(100,100),
              link_ax=None, titles=['$p$', '$\Delta p$', "$k_{est}$", '$p$', '$\Delta p$', "$k_{est}$", '', '', '' ],
              colour_functions=[propagate_BEM_pressure, propagate_BEM_laplacian_abs, '/', propagate_BEM_pressure, propagate_BEM_laplacian_abs, '/', '/','/','/'], 
              colour_function_args=[{"path":path, 'H':H, 'board':board, 'scatterer':scatterer, 'p_ref':p_ref}, {"path":path, 'H':H, 'board':board,  'scatterer':scatterer , 'p_ref':p_ref},{"post_function":check_k, 'ids':[0,1]},
                                    {"path":path, 'board':board, 'scatterer':scatterer, 'p_ref':p_ref}, {"path":path, 'board':board,  'scatterer':scatterer , 'p_ref':p_ref},{"post_function":check_k, 'ids':[3,4]},
                                    {'ids':[0,3]}, {'ids':[1,4]},{'ids':[2,5]}
                                    ],
              arrangement=(3,3),
                                    )

    # import matplotlib.pyplot as plt

    # plt.matshow(E_lap.abs().squeeze().reshape(16,16).detach())

    # plt.show20