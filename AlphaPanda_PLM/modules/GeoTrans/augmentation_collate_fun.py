import logging
import numpy as np
import torch
from schnetpack import Properties
from schnetpack.data.loader import _collate_aseatoms

# def _collate_aseatoms_Transformer_Augment(examples, translations=[1e3,1e4,1e5], kk=2):
def _collate_aseatoms_Transformer_Augment(examples, translations=[1e4], kk=2):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    prop_ = [x for x in examples[0].keys() if x[0] != '_'][0]
    # rot_mat = ortho_mat.rvs(3)
    
    idx_aug = list(range(int(len(examples) / kk),len(examples)))
    len_aug = int(len(examples) / kk)
    if kk == 1:
        idx_aug = list(range(len(examples)))
        idx_aug = list(reversed(idx_aug))
        len_aug = int(len(examples) * 0.9)
    n_atom_orig_list = [len(examples[ii]['_atomic_numbers']) for ii in range(len(examples))]
    for ii in range(len_aug):
        n_atom_orig  = n_atom_orig_list[ii]
        dist_translation = np.random.choice(translations)
        examples[ii]['_atomic_numbers'] = torch.cat(
            [examples[ii]['_atomic_numbers'],
             examples[idx_aug[ii]]['_atomic_numbers']])
        # examples[ii]['_positions'] = torch.cat(
        #     [examples[ii]['_positions'],torch.from_numpy(
        #         dist_translation + np.dot(examples[idx_aug[ii]]['_positions'],rot_mat)).float()])
        examples[ii]['_positions'] = torch.cat(
            [examples[ii]['_positions'],
                dist_translation +
                examples[idx_aug[ii]]['_positions']])
        examples[ii][prop_] = examples[ii][prop_] + \
                              examples[idx_aug[ii]][
                                  prop_]
        n_atoms = len(examples[ii]['_atomic_numbers'])
        neighborhood_idx = np.tile(
            np.arange(n_atoms, dtype=np.float32)[np.newaxis],
            (n_atoms, 1)
        )
        neighborhood_idx = neighborhood_idx[
            ~np.eye(n_atoms, dtype=np.bool)
        ].reshape(n_atoms, n_atoms - 1)
        examples[ii][Properties.neighbors] = torch.LongTensor(
            neighborhood_idx.astype(np.int))

    properties = examples[0]
    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in
        properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )
    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }

    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(
            batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0
    batch['_neighbors'] = torch.arange(0, batch['_neighbors'].shape[
        1]).expand((batch['_neighbors'].shape[0],
                    batch['_neighbors'].shape[1],
                    batch['_neighbors'].shape[1])).clone()
    batch['_neighbor_mask'] = torch.ones(batch['_neighbors'].shape)
    return batch

def _collate_aseatoms_Transformer(examples):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    batch = _collate_aseatoms(examples)
    batch['_neighbors'] = torch.arange(0, batch['_neighbors'].shape[
        1]).expand((batch['_neighbors'].shape[0],
                    batch['_neighbors'].shape[1],
                    batch['_neighbors'].shape[1])).clone() #CLONE IS BECAUSE OF PIN MEMORY
    batch['_neighbor_mask'] = torch.ones(batch['_neighbors'].shape)
    return batch


def _collate_aseatoms_Transformer_Augment_MD17(examples, transl=1e4, kk=2):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    assert False
    for ii in range(len(examples)):
        examples[ii]['_positions'] = examples[ii]['_positions'] - examples[ii]['_positions'].mean(0).unsqueeze(0)
    prop_ = [x for x in examples[0].keys() if x[0] != '_'][0]
    translations = [transl]
    # rot_mat = ortho_mat.rvs(3)

    idx_aug = list(range(len(examples) // kk,len(examples)))
    len_aug = len(examples) // kk
    if kk == 1:
        idx_aug = list(range(len(examples)))
        idx_aug = list(reversed(idx_aug))
        len_aug = int(len(examples) * 0.9)
    n_atom_orig_list = [len(examples[ii]['_atomic_numbers']) for ii in range(len(examples))]
    for ii in range(len_aug):
        n_atom_orig  = n_atom_orig_list[ii]
        dist_translation = np.random.choice(translations)
        examples[ii]['_atomic_numbers'] = torch.cat(
            [examples[ii]['_atomic_numbers'],
             examples[idx_aug[ii]]['_atomic_numbers']])
        # examples[ii]['_positions'] = torch.cat(
        #     [examples[ii]['_positions'],torch.from_numpy(
        #         dist_translation + np.dot(examples[idx_aug[ii]]['_positions'],rot_mat)).float()])
        examples[ii]['_positions'] = torch.cat(
            [examples[ii]['_positions'],
                dist_translation +
                examples[idx_aug[ii]]['_positions']])
        examples[ii][prop_] = examples[ii][prop_] + \
                              examples[idx_aug[ii]][
                                  prop_]
        if 'forces' in examples[ii].keys():
            examples[ii]['forces'] = torch.cat([examples[ii]['forces'],examples[idx_aug[ii]]['forces']],0)
        n_atoms = len(examples[ii]['_atomic_numbers'])
        neighborhood_idx = np.tile(
            np.arange(n_atoms, dtype=np.float32)[np.newaxis],
            (n_atoms, 1)
        )
        neighborhood_idx = neighborhood_idx[
            ~np.eye(n_atoms, dtype=np.bool)
        ].reshape(n_atoms, n_atoms - 1)
        examples[ii][Properties.neighbors] = torch.LongTensor(
            neighborhood_idx.astype(np.int))
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in
        properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }

    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(
            batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0
    batch['_neighbors'] = torch.arange(0, batch['_neighbors'].shape[
        1]).expand((batch['_neighbors'].shape[0],
        batch['_neighbors'].shape[1],
        batch['_neighbors'].shape[1])).clone()
    batch['_neighbor_mask'] = torch.ones(batch['_neighbors'].shape)
    return batch
