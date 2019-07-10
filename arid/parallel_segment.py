from .segment import segment
from funlib.segment.arrays import replace_values
from funlib.segment.arrays.impl import find_components
import daisy
import glob
import logging
import numpy as np
import os
import tempfile


logger = logging.getLogger(__name__)


def parallel_segment(
        embedding,
        coordinate_scale,
        threshold,
        block_size,
        num_workers,
        segmentation,
        block_segmentation=None):
    '''Create a segmentation from an embedding by growing an EMST, pruning
    edges above the given threshold, and finding connected components.

    Args:

        embedding (:class:`daisy.Array`, shape ``(k, d, h, w)``):

            A k-dimensional feature embedding of points in 3D.

        coordinate_scale(``float`` or ``tuple`` of ``float``):

            How to scale the coordinates, if used to augment the embedding.
            Should be the same as used during training.

        threshold (``float``):

            The threshold above which to prune edges from the EMST.

        segmentation (:class:`daisy.Array`, shape ``(d, h, w)``):

            The target array to store the segmentation in. It is assumed that
            all values in this array are zero when this function is called.

        block_segmentation (:class:`daisy.Array`, shape ``(d, h, w)``, optional):

            If given, store the segmentation per block in this array.
    '''

    # The context one block needs in order to produce a seamless segmentation:
    #
    # If the embedding is exactly the same (worst case), the length of the
    # longest edge in the MST is equal to the threshold. In world units, this
    # is threshold/coordinate_scale.
    #
    # However, this requires quite a large context and makes subsequent
    # connected component analysis less efficient and more involved.
    #
    # Therefore, we go for a one-voxel context here.

    context = daisy.Coordinate(embedding.voxel_size)

    write_roi = daisy.Roi(
        (0,)*len(block_size),
        block_size)
    read_roi = write_roi.grow(context, context)
    total_roi = embedding.roi.grow(context, context)

    num_voxels_in_block = (read_roi/embedding.voxel_size).size()

    if block_segmentation is None:
        block_segmentation = segmentation

    with tempfile.TemporaryDirectory() as tmpdir:

        daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: segment_in_block(
                embedding,
                block_segmentation,
                num_voxels_in_block,
                coordinate_scale,
                threshold,
                b,
                tmpdir),
            num_workers=num_workers,
            fit='shrink')

        nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    write_roi = daisy.Roi(
        (0,)*len(block_size),
        block_size)
    read_roi = write_roi
    total_roi = embedding.roi

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: relabel_in_block(
            block_segmentation,
            segmentation,
            nodes,
            components,
            b),
        num_workers=num_workers,
        fit='shrink')


def segment_in_block(
        embedding,
        segmentation,
        num_voxels_in_block,
        coordinate_scale,
        threshold,
        block,
        tmpdir):

    components = segment(
        embedding.to_ndarray(roi=block.read_roi, fill_value=0),
        embedding.voxel_size,
        coordinate_scale,
        threshold)
    components += block.block_id * num_voxels_in_block

    segmentation[block.write_roi] = components[1:-1, 1:-1, 1:-1]
    neighbors = segmentation.to_ndarray(roi=block.read_roi, fill_value=0)

    unique_pairs = []

    for d in range(3):

        slices_neg = tuple(
            slice(None) if dd != d else slice(0, 1)
            for dd in range(3)
        )
        slices_pos = tuple(
            slice(None) if dd != d else slice(-1, None)
            for dd in range(3)
        )

        pairs_neg = np.array([
            components[slices_neg].flatten(),
            neighbors[slices_neg].flatten()])
        pairs_neg = pairs_neg.transpose()

        pairs_pos = np.array([
            components[slices_pos].flatten(),
            neighbors[slices_pos].flatten()])
        pairs_pos = pairs_pos.transpose()

        unique_pairs.append(
            np.unique(
                np.concatenate([pairs_neg, pairs_pos]),
                axis=0))

    unique_pairs = np.concatenate(unique_pairs)
    zero_u = unique_pairs[:, 0] == 0
    zero_v = unique_pairs[:, 1] == 0
    non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))

    edges = unique_pairs[non_zero_filter]
    nodes = np.unique(edges)

    np.savez_compressed(
        os.path.join(tmpdir, 'block_%d.npz' % block.block_id),
        nodes=nodes,
        edges=edges)


def relabel_in_block(from_array, to_array, old_values, new_values, block):

    a = from_array.to_ndarray(block.write_roi)
    replace_values(a, old_values, new_values, inplace=True)
    to_array[block.write_roi] = a


def read_cross_block_merges(tmpdir):

    block_files = glob.glob(os.path.join(tmpdir, 'block_*.npz'))

    nodes = []
    edges = []
    for block_file in block_files:
        b = np.load(block_file)
        nodes.append(b['nodes'])
        edges.append(b['edges'])

    return np.concatenate(nodes), np.concatenate(edges)
