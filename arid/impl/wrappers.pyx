import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t


cdef extern from "connected_components.h":
    double connected_components(
        size_t numNodes,
        const double* mst,
        double threshold,
        uint64_t* components);


def find_components(
    np.ndarray[double, ndim=2] mst,
    double threshold):
    '''Find connected components in an MST under the given threshold.
    '''

    cdef size_t num_edges = mst.shape[0]
    cdef size_t num_nodes = num_edges + 1

    assert mst.shape[1] == 3, "MST not given as rows of [u, v, dist]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not mst.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous mst arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        mst = np.ascontiguousarray(mst)

    # prepare output arrays
    cdef np.ndarray[uint64_t, ndim=1] components = np.zeros(
            (num_nodes,),
            dtype=np.uint64)

    connected_components(
        num_nodes,
        &mst[0, 0],
        threshold,
        &components[0])

    return components
