import numpy as np


def compute_triangle_area_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    cross_product = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_product)


def compute_tetrahedron_volume_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    ad = points[3, :] - points[0, :]
    # Calculate the scalar triple product
    volume = abs(np.dot(np.cross(ab, ac), ad)) / 6
    return volume


def compute_measure_per_elem_(points, elem_dim):
    '''
    Compute element measure (length, area or volume)
    for 2-point  element, compute its length
    for 3-point  element, compute its area
    for 4-point  element, compute its area if elem_dim=2; compute its volume if elem_dim=3
    equally assign it to its nodes

        Parameters: 
            points : float[npoints, ndims]
            elem_dim : int

        Returns:
            s : float
    '''

    npoints, ndims = points.shape
    if npoints == 2:
        s = np.linalg.norm(points[0, :] - points[1, :])
    elif npoints == 3:
        s = compute_triangle_area_(points)
    elif npoints == 4:
        assert (npoints == 3 or npoints == 4)
        if elem_dim == 2:
            s = compute_triangle_area_(
                points[:3, :]) + compute_triangle_area_(points[1:, :])
        elif elem_dim == 3:
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("elem dim ", elem_dim, "is not recognized")
    else:
        raise ValueError("npoints ", npoints, "is not recognized")
    return s


def compute_node_measures(nodes, elems):
    '''
    Compute node measures  (separate length, area and volume ... for each node), 
    For each element, compute its length, area or volume s, 
    equally assign it to its ne nodes (measures[:] += s/ne).

        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.

        Return :
            measures : float[nnodes, nmeasures]
                       padding NaN for nodes that do not have measures
                       nmeasures >= 1: number of measures with different dimensionalities
                       For example, if there are both lines and triangles, nmeasures = 2

    '''
    nnodes, ndims = nodes.shape
    measures = np.full((nnodes, ndims), np.nan)
    measure_types = [False] * ndims
    for elem in elems:
        elem_dim, e = elem[0], elem[1:]
        e = e[e >= 0]
        ne = len(e)
        # compute measure based on elem_dim
        s = compute_measure_per_elem_(nodes[e, :], elem_dim)
        # assign it to cooresponding measures
        measures[e, elem_dim - 1] = np.nan_to_num(measures[e, elem_dim - 1], nan=0.0)
        measures[e, elem_dim - 1] += s / ne
        measure_types[elem_dim - 1] = True

    # return only nonzero measures
    return measures[:, measure_types]


def compute_node_weights(nnodes, node_measures, equal_weights=False):
    '''
    This function calculates weights and rhos for each node using its corresponding measures.
            v(x) = ∫ u(x)rho(x) dx 
                 ≈ ∑ u(x_i)rho(x_i)m(x_i) 
                 = ∑ u(x_i)w(x_i)
            w(x_i) = rho(x_i)m(x_i)
    Node weights are computed such that their sum equals 1, because:
              1  = ∫ rho(x) dx 
                 ≈ ∑ rho(x_i)m(x_i)
                 = ∑ w(x_i)
    If there are several types of measures, compute weights for each type of measures, and normalize it by nmeasures
            w_k(x_i)= w_k(x_i)/nmeasures 

    Parameters:
        nnodes int[ndata]: 
            Number of nodes for each data instance.

        node_measures float[ndata, max_nnodes, nmeasures]: 
            Each value corresponds to the measure of a node.
            Padding with NaN is used for indices greater than or equal to the number of nodes (`nnodes`), or nodes do not have measure

        equal_weights bool:
            - True
                    w(x_i)=1/nnodes
              and we can recover rho by
                    rho(x_i) = w(x_i)/m(x_i) = 1/(m(x_i)*nnodes)
            - False 
                    rho(x_i)=1/|Omega|
               and we can compute w by
                    w(x_i) = rho(x_i)m(x_i) = m(x_i)/|Omega|

    Returns:
        node_weights float[ndata, max_nnodes, nmeasures]: 
            Array of computed node weights, maintaining the same padding structure.
        node_rhos   float[ndata, max_nnodes, nmeasures]: 
            Array of computed node rhos, maintaining the same padding structure.
    '''

    ndata, max_nnodes, nmeasures = node_measures.shape
    node_weights = np.zeros((ndata, max_nnodes, nmeasures))
    node_rhos = np.zeros((ndata, max_nnodes, nmeasures))
    if equal_weights:
        for i in range(ndata):
            n = nnodes[i]
            for j in range(nmeasures):
                # take average for nonzero measure nodes
                S = sum(node_measures[i, :n, j])
                node_weights[i, :n, j] = 1 / n
                node_rhos[i, :n, j] = 1 / (node_measures[i, :n, j] * n)

    else:
        for i in range(ndata):
            n = nnodes[i]
            for j in range(nmeasures):
                S = sum(node_measures[i, :n, j])
                node_rhos[i, :n, j] = 1 / S
                node_weights[i, :n, j] = node_measures[i, :n, j] / S

    node_weights = node_weights / nmeasures
    node_rhos = node_rhos / nmeasures

    return node_weights, node_rhos
