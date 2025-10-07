import numpy as np
from math import prod


def convert_structured_data(coords_list, features, nnodes_per_elem=3, feature_include_coords=True):
    '''
    Convert structured data, to unstructured data, support both 2d and 3d coordinates
    coords_list stores x, y, (z) coordinates of each points in list of ndims float[nnodes, nx, ny, (nz)], coords_list[i] is as following
                    nz-1       ny-1                                                         
                    nz-2     ny-2                                                            
                    .       .                                                               
    z direction     .      .   (y direction)            
                    .     1                                                                 
                    1    .                                                                    
                    0   0                                                                    
                        0 - 1 - 2 - ... - nx-1   (x direction)
    For example, it can be generated as
    grid_1d_x, grid_1d_y, grid_1d_z = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), np.linspace(0, Lz, nz)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d_x, grid_1d_y, grid_1d_z, indexing="ij")
    coords_list = [grid_x, grid_y, grid_z]

    Then we order the nodes by iterating z, then y, then x (reshape)
    for i in range(nx-1): 
        for j in range(ny-1): 
            for k in range(nz-1): 
                id = i*ny*nz + j*nz + k
    For example when nx=ny=nz, the ordering is as following
          3 -------- 7
         /|         /|
        1 -------- 5 |
        | |        | |
      z | 2 -------|-6
        |/y        |/
        0 ----x----4      


        Parameters:  
            coords_list            :  list of ndims float[nnodes, nx, ny, (nz)], for each dimension coords_list[0], coords_list[1],... are x, y,... coordinates
            features               :  float[nelems, nx, ny, (nz), nfeatures], features on each point
            nnodes_per_elem        :  int, describing element type
                                      nnodes_per_elem = 3: 2d triangle mesh; 
                                      nnodes_per_elem = 4: 2d quad mesh or 3d tetrahedron mesh
                                      nnodes_per_elem = 8: 3d hexahedron mesh
            feature_include_coords :  boolean, whether treating coordinates as features, if coordinates
                                      are treated as features, they are concatenated at the end

        Return :  
            nodes_list :     list of float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            features_list  : list of float[nnodes, nfeatures]
    '''
    ndims = len(coords_list)
    print("convert_structured_data for ", ndims, " problems")
    ndata, *dims = coords_list[0].shape
    nnodes = prod(dims)
    # construct nodes
    nodes = np.stack([coords_list[i].reshape((ndata, nnodes)) for i in range(ndims)], axis=2)
    # construct features
    if feature_include_coords:
        nfeatures = features.shape[-1] + ndims
        features = np.concatenate((features.reshape((ndata, nnodes, -1)), nodes), axis=-1)
    else:
        nfeatures = features.shape[-1]
        features = features.reshape((ndata, nnodes, -1))

    # construct elements
    if (ndims == 2):  # triange (nnodes_per_elem = 3), quad (nnodes_per_elem = 4)
        assert (nnodes_per_elem == 4 or nnodes_per_elem == 3)
        nx, ny = dims
        nelems = (nx - 1) * (ny - 1) * (1 if nnodes_per_elem == 4 else 2)
        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx - 1):
            for j in range(ny - 1):
                ie = i * (ny - 1) + j  # element id
                # node ids clockwise
                #   1 ---------2
                #  /          /
                # 0 -------- 3
                ins = [i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j]
                if nnodes_per_elem == 4:
                    elems[ie, :] = 2, ins[0], ins[1], ins[2], ins[3]
                else:
                    elems[2 * ie, :] = 2, ins[0], ins[1], ins[2]
                    elems[2 * ie + 1, :] = 2, ins[0], ins[2], ins[3]
    elif (ndims == 3):  # tetrahedron (nnodes_per_elem = 4), cubic (nnodes_per_elem = 8)
        assert (nnodes_per_elem == 8 or nnodes_per_elem == 4)
        nx, ny, nz = dims
        nelems = (nx - 1) * (ny - 1) * (nz - 1) * (1 if nnodes_per_elem == 8 else 6)
        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    ie = i * (ny - 1) * (nz - 1) + j * (nz - 1) + k  # element id
                    # node ids for k, and k+1 in counterclockwise
                    #   7 -------- 6
                    #  /|         /|
                    # 4 -------- 5 |
                    # | |        | |
                    # | 3 -------|-2
                    # |/         |/
                    # 0 -------- 1
                    ins = [i * ny * nz + j * nz + k, (i + 1) * ny * nz + j * nz + k, (i + 1) * ny * nz + (j + 1) * nz + k, i * ny * nz + (j + 1) * nz + k,
                           i * ny * nz + j * nz + (k + 1), (i + 1) * ny * nz + j * nz + (k + 1), (i + 1) * ny * nz + (j + 1) * nz + (k + 1), i * ny * nz + (j + 1) * nz + (k + 1)]
                    if nnodes_per_elem == 8:
                        elems[ie, :] = 3, ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7]
                    else:
                        elems[6 * ie, :] = 3, ins[0], ins[1], ins[3], ins[5]
                        elems[6 * ie + 1, :] = 3, ins[0], ins[3], ins[5], ins[7]
                        elems[6 * ie + 2, :] = 3, ins[0], ins[4], ins[5], ins[7]
                        elems[6 * ie + 3, :] = 3, ins[1], ins[2], ins[3], ins[5]
                        elems[6 * ie + 4, :] = 3, ins[2], ins[5], ins[6], ins[7]
                        elems[6 * ie + 5, :] = 3, ins[2], ins[3], ins[5], ins[7]

    elems = np.tile(elems, (ndata, 1, 1))
    nodes_list = [nodes[i, ...] for i in range(ndata)]
    elems_list = [elems[i, ...] for i in range(ndata)]
    features_list = [features[i, ...] for i in range(ndata)]

    return nodes_list, elems_list, features_list
