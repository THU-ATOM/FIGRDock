# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from rdkit import Chem

from rdkit.Chem import AllChem

import numpy as np
import contextlib
# from torchdrug import data as td
from torch_geometric.utils import dense_to_sparse

import torch

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat


# mol_mask[i][j] = 1 means that atom i and atom j are  
# connected by a bond(origin adjacent matrix), or 2-hop away, or in the same ring structure
def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        if len(ring) > 8:
            continue
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask


def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False, ):
    # N x 3
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        # N x N
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
        LAS_edge_index, _ = dense_to_sparse(LAS_distance_constraint_mask)
    else:
        LAS_distance_constraint_mask = None
        LAS_edge_index = None
    
    return LAS_edge_index
    # molstd = td.Molecule.from_molecule(mol, node_feature='property_prediction')
    # # molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')
    # compound_node_features = molstd.node_feature # nodes_chemical_features
    # edge_list = molstd.edge_list # [num_edge, 3] (node_in, node_out, relation)
    # edge_weight = molstd.edge_weight # [num_edge, 1]
    # assert edge_weight.max() == 1
    # assert edge_weight.min() == 1
    # assert coords.shape[0] == compound_node_features.shape[0]
    # x = [torch.tensor(coords), compound_node_features, edge_list, LAS_edge_index]
    # return x
