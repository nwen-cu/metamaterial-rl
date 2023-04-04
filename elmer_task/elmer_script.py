def run(data=None):
    import os
    import sys
    from datetime import datetime
    from pathlib import Path
    import logging
    
    import numpy as np
    
    # Use test data if no data provided
    if not data:
        angle_matrix = np.array([[-80, -80, -45, -45, 45, 45, 80, 80], 
                                [-80, -80, -45, -45, 45, 45, 80, 80],
                                [-45, -45, -10, -10, 10, 10, 45, 45],
                                [-45, -45, -10, -10, 10, 10, 45, 45],
                                [45, 45, 10, 10, -10, -10, -45, -45],
                                [45, 45, 10, 10, -10, -10, -45, -45],
                                [80, 80, 45, 45, -45, -45, 80, 80],
                                [80, 80, 45, 45, -45, -45, 80, 80]])

        data = {
            'input': {
                'angle_matrix': angle_matrix,
            },
            'task': {
                'work_dir': Path('/home/nwen/simhub/client/test_work_dir').absolute(),
                'data_dir': Path('/home/nwen/simhub/client/elmer_task/data').absolute(),
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

    mesh_size = 2.

    def create_unit(rotation, offset, unit_size=(40., 40.), frame_width=1, stripe_width=2., stripe_offset=0., ignore_mass=1.):
        unit_diag = np.sqrt(unit_size[0] ** 2 + unit_size[1] ** 2)
        stripe_offset %= stripe_width
        max_stripe_num = int(unit_diag / (stripe_width))

        rotation = np.deg2rad(rotation)

        stripe_dimtags = []
        for i in range(-1, max_stripe_num + 1):
            stripe_tag = factory.add_rectangle(0, i * stripe_width * 2 + stripe_offset, 0, unit_diag, stripe_width)
            stripe_dimtags.append((2, stripe_tag))

        factory.translate(stripe_dimtags, -unit_diag / 2, -unit_diag / 2, 0.)

        factory.rotate(stripe_dimtags, 0., 0., 0., 0., 0., 1., rotation)

        stripe_mask_tag = factory.add_rectangle(frame_width - unit_size[0] / 2, frame_width - unit_size[1] / 2, 0, 
                                                unit_size[0] - frame_width * 2, unit_size[1] - frame_width * 2)

        infill_dimtags, _ = factory.intersect([(2, stripe_mask_tag)], stripe_dimtags)

        remove_dimtags = []
        for dimtag in infill_dimtags:
            if factory.get_mass(*dimtag) < ignore_mass:
                remove_dimtags.append(dimtag)

        for dimtag in remove_dimtags:
            infill_dimtags.remove(dimtag)

        factory.remove(remove_dimtags)

        factory.synchronize()

        factory.translate(infill_dimtags, unit_size[0] / 2 + offset[0], unit_size[1] / 2 + offset[1], 0.)

        factory.synchronize()

        return infill_dimtags

    units = data['input']['angle_matrix']

    unit_size = (40, 40)
    board_margin = (10, 10)

    units = np.flip(np.array(units).T, axis=1)
    unit_size = np.array(unit_size)
    board_margin = np.array(board_margin)
    board_size = units.shape * unit_size + board_margin * 2

    board_tag = factory.add_rectangle(0, 0, 0, *board_size)

    stripe_dimtags = []

    it = np.nditer(units, flags=['multi_index'])
    for unit in it:
        offset = it.multi_index * unit_size + board_margin
        unit_dimtags = create_unit(unit, offset)
        stripe_dimtags.extend(unit_dimtags)

    factory.synchronize()

    frame_dimtags, _ = factory.cut_fragment([(2, board_tag)], stripe_dimtags)

    factory.synchronize()

    frame_tags = [dimtag[1] for dimtag in frame_dimtags]
    infill_tags = [dimtag[1] for dimtag in stripe_dimtags]

    phy_frame = add_physical_group(2, frame_tags, "frame")
    phy_infill = add_physical_group(2, infill_tags, "infill")

    left_bound = get_boundaries_in_box(0., 0., 0., 0., board_size[1], 0., 2, frame_dimtags[0][1])
    right_bound = get_boundaries_in_box(board_size[0], 0., 0., board_size[0], board_size[1], 0., 2, frame_dimtags[0][1])

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
    elmer.data_dir = data['task']['data_dir']

    sim = elmer.load_simulation("2D_steady")

    pdms = elmer.load_material("PDMS", sim)
    copper = elmer.load_material("copper", sim)

    solver_heat = elmer.load_solver("HeatSolver", sim)
    solver_output = elmer.load_solver("ResultOutputSolver", sim)
    eqn = elmer.Equation(sim, "main", [solver_heat])

    T0 = elmer.InitialCondition(sim, "T0", {"Temperature": 273.15})

    body_frame = elmer.Body(sim, "frame", [phy_frame])
    body_frame.material = copper
    body_frame.initial_condition = T0
    body_frame.equation = eqn

    body_infill = elmer.Body(sim, "infill", [phy_infill])
    body_infill.material = pdms
    body_infill.initial_condition = T0
    body_infill.equation = eqn

    boundary_left = elmer.Boundary(sim, "left_bound", [phy_left])
    boundary_left.data.update({"Temperature": 353.15})  # 80 °C
    boundary_right = elmer.Boundary(sim, "right_bound", [phy_right])
    boundary_right.data.update({"Temperature": 293.15})  # 20 °C

    sim.write_startinfo(str(workdir))
    sim.write_sif(str(workdir))

    ##############
    # execute ElmerGrid & ElmerSolver
    execute.run_elmer_grid(str(workdir), "case.msh")
    execute.run_elmer_solver(str(workdir))
    
    import subprocess
    with open(data['task']['work_dir'] / 'debug.log', 'w') as fp:
        sp = subprocess.run('ElmerSolver case.sif', shell=True, stdout=fp, stderr=fp)
        fp.write(f'exit({sp.returncode})\n')
        
        sp = subprocess.run('module list', shell=True, stdout=fp, stderr=fp)
        fp.write(f'exit({sp.returncode})\n')
        
        sp = subprocess.run('lscpu', shell=True, stdout=fp, stderr=fp)
        fp.write(f'exit({sp.returncode})\n')
        
        sp = subprocess.run('env', shell=True, stdout=fp, stderr=fp)
        fp.write(f'exit({sp.returncode})\n')
    
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
    
    sampling_size = units.shape * np.array(10)

    extraction_resolution = board_size / sampling_size

    m = pv.read(str(workdir / 'case_t0001.vtu'))
    m.set_active_scalars('temperature')

    kdtree = cKDTree(m.points.astype(np.double))

    sampling_mesh = np.mgrid[0.:board_size[0]:extraction_resolution[0], 0.:board_size[1]:extraction_resolution[1]]
    sampling_spot = sampling_mesh.T.reshape((sampling_mesh.shape[1] * sampling_mesh.shape[2], 2))
    dist, index = kdtree.query(np.hstack((sampling_spot, np.zeros((sampling_spot.shape[0], 1), dtype=sampling_spot.dtype))))

    values = m.active_scalars[index]
    result = np.hstack([sampling_spot, values.reshape((values.shape[0], 1))]).T
    
    print('Spatial data extracted')

    extract_time = datetime.now() - start_time
    extract_time = extract_time.total_seconds()
    
    data['output'] = {
        'temperature_distribution': result,
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
    
    result = run(None)
    
    pprint(result)
    