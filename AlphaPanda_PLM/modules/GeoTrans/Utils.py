import torch

def atom_distances(
        positions,
        neighbors,
        cell=None,
        cell_offsets=None,
        return_vecs=False,
        normalize_vecs=False,
        neighbor_mask=None,
        inverse_flag=True):
    r"""Compute distance of every atom to its neighbors.

    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Args:
        positions (torch.Tensor):
            atomic Cartesian coordinates with (N_b x N_at x 3) shape
        neighbors (torch.Tensor):
            indices of neighboring atoms to consider with (N_b x N_at x N_nbh) shape
        cell (torch.tensor, optional):
            periodic cell of (N_b x 3 x 3) shape
        cell_offsets (torch.Tensor, optional) :
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        return_vecs (bool, optional): if True, also returns direction vectors.
        normalize_vecs (bool, optional): if True, normalize direction vectors.
        neighbor_mask (torch.Tensor, optional): boolean mask for neighbor positions.

    Returns:
        (torch.Tensor, torch.Tensor):
            distances:
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec:
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).

    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device,
                         dtype=torch.long)[
            :, None, None
            ]
    # Get atomic positions of all neighboring indices
    if neighbors is not None:
        pos_xyz = positions[idx_m, neighbors[:, :, :], :]
    else:
        pos_xyz = positions[idx_m, :, :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    distances = torch.norm(dist_vec, 2, 3)
    if inverse_flag:
        distances = 1. / (distances + float(1e-8))
        #distances = 1. / (distances + float(1e-16)) #huyue
        #if not torch.isfinite(distances):
        if float('inf') in torch.isfinite(distances) or float('-inf')  in torch.isfinite(distances) or float('nan') in torch.isfinite(distances):
            print('distances is NaN or Inf detected.')
            print(distances)
            print('\n')

    else:
        distances =  distances +0.
    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        #         tmp_distances = torch.zeros_like(distances)
        #         tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        #         distances = tmp_distances
        tmp = neighbor_mask.float()
        neighbor_mask = torch.bmm(tmp.unsqueeze(2), tmp.unsqueeze(1)).bool()
        # if inverse_flag:
        distances.diagonal(dim1=-2, dim2=-1)[:] = 0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[
            neighbor_mask != 0]
        distances = tmp_distances
    # print(distances[0])
    # exit(1)
    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[
            neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    # print(distances)
    return distances


class AtomDistances(torch.nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        return_directions (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(
            self, positions, neighbors, cell=None, cell_offsets=None,
            neighbor_mask=None, inverse_flag=True,mahalanobis_mat=None
    ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.
            :param inverse_flag:

        """
        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask, inverse_flag=inverse_flag
        )
