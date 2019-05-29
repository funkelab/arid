from funlib.learn.tensorflow.losses import get_emst
from .impl import find_components
import numpy as np

def segment(embedding, voxel_size, coordinate_scale, threshold):
    '''Create a segmentation from an embedding by growing an EMST, pruning
    edges above the given threshold, and finding connected components.

    Args:

        embedding (``ndarray``, shape ``(k, d, h, w)``):

            A k-dimensional feature embedding of points in 3D.

        voxel_size (``tuple`` of ``float``):

            The voxel size of the embedding.

        coordinate_scale(``float`` or ``tuple`` of ``float``):

            How to scale the coordinates, if used to augment the embedding.
            Should be the same as used during training.

        threshold (``float``):

            The threshold above which to prune edges from the EMST.

    Returns:

        ``ndarray`` of size ``(d, h, w)`` containing the segmentation.
    '''

    try:
        coordinate_scale = tuple(coordinate_scale)
    except:
        coordinate_scale = (coordinate_scale,)*3

    k, depth, height, width = embedding.shape

    assert len(voxel_size) == 3
    assert len(coordinate_scale) == 3

    scale = tuple(c*s for c, s in zip(coordinate_scale, voxel_size))

    coordinates = np.meshgrid(
        np.arange(0, depth*scale[0], scale[0]),
        np.arange(0, height*scale[1], scale[1]),
        np.arange(0, width*scale[2], scale[2]),
        indexing='ij')
    for i in range(len(coordinates)):
        coordinates[i] = coordinates[i].astype(np.float32)
    embedding = np.concatenate([embedding, coordinates], 0)
    embedding = np.transpose(embedding, axes=[1, 2, 3, 0])
    embedding = np.reshape(embedding, [depth*width*height, -1])

    emst = get_emst(embedding)

    components = find_components(emst, threshold)

    return components.reshape((depth, height, width))
