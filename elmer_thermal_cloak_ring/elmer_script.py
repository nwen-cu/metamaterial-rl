def run(data=None):
    import os
    import sys
    from datetime import datetime
    from pathlib import Path
    import logging
    
    import numpy as np
    
    # Use test data if no data provided
    if not data:
        data = {
            'input': {
                'r': [50., 80., 100.],
                
                # 'k': [0.15, 1.5, 51.9, 12.],
                'k': [12., 12., 12., 12.],
            },
            'task': {
                'work_dir': Path('/home/nwen/metamaterial-rl/elmer_harv/test_workdir').absolute(),
                'data_dir': Path('/home/nwen/metamaterial-rl/elmer_harv').absolute(),
            },
            'metrics': {
                'submit_timestamp': datetime.now(),
            }
        }
     
    workdir = data['task']['work_dir']

    import gmsh
    import pyvista as pv
    
    from scipy.spatial import cKDTree

    from pyelmer import elmer
    from pyelmer import execute
    from pyelmer.post import scan_logfile
    from objectgmsh.utils import add_physical_group, get_boundaries_in_box

    wait_time = (datetime.now() - data['metrics']['submit_timestamp']).total_seconds()
        
    def cut_fragment(objectDimTags, toolDimTags):
        import gmsh
        factory = gmsh.model.occ

        cutDimTags, _ = factory.cut(objectDimTags, toolDimTags, removeTool=False)
        fragDimTags, _ = factory.fragment(cutDimTags, toolDimTags)
        return cutDimTags, fragDimTags

    gmsh.model.occ.cut_fragment = cut_fragment

    ###############
    os.environ['OMP_NUM_THREADS'] = '1'

    start_time = datetime.now()

    gmsh.initialize()
    
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("heat-transfer-2d")
    factory = gmsh.model.occ

    mesh_size = 4.


    board_size = (400., 400.)
    board_center = (board_size[0] / 2, board_size[1] / 2)

    # Region 0~2, 0 is the center circle, 1 is ring in the middle, 2 is the outer ring, 3/board is the remaining of the board
    # r[0~2] is radius of the outer boundry, k[0~2] is the conductivity 
    r = list(data['input']['r'])
    k = list(data['input']['k'])
    
    for i in range(len(k)):
        k[i] = max(0.15, k[i])
    
    region_tags = []

  
    board_tag = factory.add_rectangle(0, 0, 0, *board_size)
    
    cuttee_tag = board_tag
    
    for i in range(len(r) - 1, -1, -1):
        cutter_tag = factory.add_circle(board_size[0] / 2, board_size[1] / 2, 0, r[i])
        cutter_tag = factory.add_curve_loop([cutter_tag])
        cutter_tag = factory.add_plane_surface([cutter_tag])
        
        cuttee_tag, _ = factory.cut_fragment([(2, cuttee_tag)], [(2, cutter_tag)])
        cuttee_tag = cuttee_tag[0][1]
        
        region_tags.append(cuttee_tag)
        
        cuttee_tag = cutter_tag
        
    region_tags.append(cutter_tag)
    
    region_tags.reverse()

    factory.synchronize()

    
    phy_regions = []
    
    for i in range(len(k)):
        phy_region = add_physical_group(2, [region_tags[i]], f"region{i}")
        phy_regions.append(phy_region)

    left_bound = get_boundaries_in_box(0., 0., 0., 0., board_size[1], 0., 2, region_tags[-1])
    right_bound = get_boundaries_in_box(board_size[0], 0., 0., board_size[0], board_size[1], 0., 2, region_tags[-1])

    phy_left = add_physical_group(1, [left_bound], "left_bound")
    phy_right = add_physical_group(1, [right_bound], "right_bound")

    # create mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(workdir / 'case.msh'))
    
    mesh_retries = 2
    
    while (not (workdir / 'case.msh').exists()) or (workdir / 'case.msh').stat().st_size < 500:
        if mesh_retries > 0:
            print('Meshing failed, retrying...')
            gmsh.model.mesh.generate(2)
            gmsh.write(str(workdir / 'case.msh'))
            mesh_retries -= 1
        else:
            raise ArithmeticError('Failed to mesh the geometries')
            
    print('Mesh generated')

    mesh_time = datetime.now() - start_time
    mesh_time = mesh_time.total_seconds()

    start_time = datetime.now()

    ###############
    # elmer setup
    elmer.data_dir = str(Path(data['task']['data_dir']) / 'data')

    sim = elmer.load_simulation("2D_steady")

    solver_heat = elmer.load_solver("HeatSolver", sim)
    solver_output = elmer.load_solver("ResultOutputSolver", sim)
    eqn = elmer.Equation(sim, "main", [solver_heat])

    T0 = elmer.InitialCondition(sim, "T0", {"Temperature": 273.15})
    
    body_regions = []
    
    for i in range(len(phy_regions)):
        body_region = elmer.Body(sim, f"region{i}", [phy_regions[i]])
        body_region.material = elmer.Material(sim, f'm{i}', {'Heat Conductivity': k[i]})
        body_region.initial_condition = T0
        body_region.equation = eqn

    boundary_left = elmer.Boundary(sim, "left_bound", [phy_left])
    boundary_left.data.update({"Temperature": 373.15})  # 100 °C
    boundary_right = elmer.Boundary(sim, "right_bound", [phy_right])
    boundary_right.data.update({"Temperature": 293.15})  # 20 °C

    sim.write_startinfo(str(workdir))
    sim.write_sif(str(workdir))

    ##############
    # execute ElmerGrid & ElmerSolver
    execute.run_elmer_grid(str(workdir), "case.msh")
    execute.run_elmer_solver(str(workdir))
    
    print('FEM solved')
      
    
    ###############
    # scan log for errors and warnings
    err, warn, stats = scan_logfile(str(workdir))
    print("Errors:", err)
    print("Warnings:", warn)
    print("Statistics:", stats)

    solve_time = datetime.now() - start_time
    solve_time = solve_time.total_seconds()

    start_time = datetime.now()

    extraction_resolution = (1, 1)

    m = pv.read(str(workdir / 'case_t0001.vtu'))
    m.set_active_scalars('temperature')

    kdtree = cKDTree(m.points.astype(np.double))

    sampling_mesh = np.mgrid[0.:board_size[0]:extraction_resolution[0], 0.:board_size[1]:extraction_resolution[1]]
    sampling_spot = sampling_mesh.T.reshape((sampling_mesh.shape[1] * sampling_mesh.shape[2], 2))
    dist, index = kdtree.query(np.hstack((sampling_spot, np.zeros((sampling_spot.shape[0], 1), dtype=sampling_spot.dtype))))

    values = m.active_scalars[index]
    result = np.hstack([sampling_spot, values.reshape((values.shape[0], 1))]).T
    
#     ta = m.active_scalars[kdtree.query([  -r[-1] + 200.,   200.,   0.])[1]]
#     tb = m.active_scalars[kdtree.query([  -r[0] + 200.,   200.,   0.])[1]]
#     tc = m.active_scalars[kdtree.query([  r[0] + 200.,   200.,   0.])[1]]
#     td = m.active_scalars[kdtree.query([  r[-1] + 200.,   200.,   0.])[1]]
    
#     delta_t = np.abs(tb - tc)
    
    print('Spatial data extracted')

    extract_time = datetime.now() - start_time
    extract_time = extract_time.total_seconds()
    
    data['output'] = {
        'temperature_distribution': result,
        # 'temp_abcd': (ta, tb, tc, td),
        # 'delta_t': delta_t,
    }
    data['output_files'] = {}
    data['metrics'] |= {
        'wait_time': wait_time,
        'mesh_time': mesh_time,
        'solve_time': solve_time,
        'extract_time': extract_time,
        'mesh_retries': 2 - mesh_retries,
    }

    return data


if __name__ == '__main__':
    from pprint import pprint
    import datajson
    
    result = run(None)
    
    pprint(result)
    
    with open('output.json', 'w') as fp:
        fp.write(datajson.dump_json(result['output']))