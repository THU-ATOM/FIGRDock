import lmdb
import re
import pickle
import networkx as nx
import copy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch_geometric.data import Dataset, HeteroData
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs

from scipy import spatial
import scipy.spatial as spa
from scipy.special import softmax
from collections import defaultdict

SORTING_DICT = {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS": ["N", "CA", "C", "O", "CB", "SG"],
        "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY": ["N", "CA", "C", "O"],
        "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
        "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER": ["N", "CA", "C", "O", "CB", "OG"],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    }

def check_res_intact(res_dict: dict, complex_name=None) -> dict or None:
    SORTING_DICT = {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS": ["N", "CA", "C", "O", "CB", "SG"],
        "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY": ["N", "CA", "C", "O"],
        "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
        "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER": ["N", "CA", "C", "O", "CB", "OG"],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    }

    atoms = res_dict["atoms"]
    restype = res_dict["restypes"][0]

    if restype not in SORTING_DICT:
        return None
    
    intact_list = SORTING_DICT[restype]
    if atoms == intact_list:
        return res_dict
    elif set(intact_list).issubset(set(atoms)):
        filtered_res_dict = defaultdict(list)
        for atom_intact in intact_list:
            for i, atom in enumerate(res_dict["atoms"]):
                if atom == atom_intact:
                    filtered_res_dict["atoms"].append(atom)
                    filtered_res_dict["restypes"].append(res_dict["restypes"][i])
                    filtered_res_dict["restype_tokens"].append(res_dict["restype_tokens"][i])
                    filtered_res_dict["resnums"].append(res_dict["resnums"][i])
                    filtered_res_dict["coords"].append(res_dict["coords"][i])
                    filtered_res_dict["holo_coords"].append(res_dict["holo_coords"][i])
                    if complex_name is not None:
                        filtered_res_dict["lm_embeddings"].append(res_dict["lm_embeddings"][i])

        if filtered_res_dict["atoms"] == intact_list:
            return filtered_res_dict
        else:
            return None
    else:
        return None

def filter_side_chain_atoms(atom):
    # ignores the O-H, OXT and NH2 group and drops the H atoms
    # re returns no match if we should keep the atom for the
    # side chain torsion graph
    return re.search("^(OXT)$|^C$|^O$|^N$|^H|^H$.|^H.$[1-9]",atom) is None

def add_edges(G):
    # add edges according to logic -> connect all heavy atoms in correct order
    orderDict = {"A":"B","B":"G","G":"D","D":"E","E":"Z","Z":"H","H":""}
    atoms = list(G.nodes)
    for i in range(len(G.nodes)-1):
        for j in range(i+1,len(G.nodes),1):
            cur,_next = atoms[i], atoms[j]
            # handle special 5- ring connections for his,trp
            if (cur,_next) == ("CE1","NE2") or (cur,_next) == ("NE1","CE2") or (cur,_next) == ("CD2","CE3") or (cur,_next) == ("CZ3","CH2"):
                    G.add_edge(cur,_next)
            # if both have length 3 we have to match number and identification char
            # i.e. mactch CD2 -> CE2 , but not CD1 -> CE2
            if len(cur) == len(_next) == 3:
                if orderDict[cur[1]] == _next[1] and cur[2]==_next[2]:
                    # match, e.g. CD1 -> CE1
                    G.add_edge(cur,_next)
            else:
                if orderDict[cur[1]] == _next[1]:
                    # match, e.g. CD -> CE1
                    G.add_edge(cur,_next)


# def get_sidechain_rotation_mask(residue,flexResIndexFullAtoms, true_res = None) :
def get_sidechain_rotation_mask(atom_names, flexResIndexFullAtoms, true_res = None) :
    #compute rotatable bonds and rotation mask for each rotatable bond 

    # filter out non-heavy atoms
    # nodes = list(filter(filter_side_chain_atoms,[atom.name for atom in residue.child_list]))
    # # get mask that maps index of nodes to residue index 
    # heavy_atoms_mask = [i for i, atom in enumerate(residue.get_atoms()) if atom.name in nodes]
    
    
    nodes = list(filter(filter_side_chain_atoms, atom_names))
    # get mask that maps index of nodes to residue index 
    heavy_atoms_mask = [i for i, atom in enumerate(atom_names) if atom in nodes]
    if true_res is not None:

        rows=[]
        coords=[]

    # build graph 
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    add_edges(G)

    mask_rotate = []
    # traverse the side chain graph from the CA downwards
    # to find rotatable bonds in correct order
    for i,edge in enumerate(nx.bfs_tree(G, "CA").edges()):
        G2 = G.to_undirected()
        G2.remove_edge(*edge)
        if not nx.is_connected(G2):
            # possible rotatable bond
            
            # find subcomponent that has second vertex of current edge in it 
            # -> this is the one that will get rotated
            connectedComponents = list(nx.connected_components(G2))
            
            # find subcomponent that contains second vertex of edge
            # which is the one to be rotated
            for k,component in enumerate(connectedComponents):
                if edge[1] in component:
                    idx = k
                    break
            if len(connectedComponents[idx]) > 1:
                G2Nodes = list(G2.nodes)
                # rotate this subcomponent
                rotComponent = []
                for j,vertex in enumerate(connectedComponents[idx]):
                    # map graph idx to residue atom index and the residue atom index to global 
                    # atom index by the flexResIndexFullAtoms offset 
                    complexGraphIdx = heavy_atoms_mask[G2Nodes.index(vertex)] + flexResIndexFullAtoms
                    rotComponent.append(complexGraphIdx)
                    if true_res is not None:
                        rows.append(complexGraphIdx)
                        coords.append(true_res[nodes[G2Nodes.index(vertex)]].get_coord())
                # (subcomponentToRotate,EdgeToRotateAround)
                mask_rotate.append((rotComponent,[heavy_atoms_mask[G2Nodes.index(edge[0])] + flexResIndexFullAtoms,heavy_atoms_mask[G2Nodes.index(edge[1])] + flexResIndexFullAtoms]))
    
    return_dict = {"subcomponents":[m[0] for m in mask_rotate],
                   "edge_idx":[m[1] for m in mask_rotate]}
     
    if true_res is not None:
        return_dict["rows"] = rows
        return_dict["coords"] = coords

    return return_dict




def modify_sidechain_torsion_angle(pos, edge_index, mask_subcomponent, subcomponents, torsion_update, as_numpy=False):
    # modify single sidechain torsion angle 
    pos = copy.deepcopy(pos)
    orig_device = None
    if type(pos) != np.ndarray:
        orig_device = pos.device
        pos = pos.cpu().numpy()

    assert len(edge_index) == 2 # make sure that its just a single bond
    if torsion_update != 0:
        u, v = edge_index[0], edge_index[1]
        mask_rotate = subcomponents[mask_subcomponent[0]:mask_subcomponent[1]]
        if type(mask_rotate) != np.ndarray: mask_rotate = mask_rotate.cpu().numpy()
        try:
            rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
            rot_vec = rot_vec * torsion_update / np.linalg.norm(rot_vec)  # idx_edge!
            rot_mat = R.from_rotvec(rot_vec).as_matrix()
            pos[mask_rotate] = (pos[mask_rotate] - pos[v]) @ rot_mat.T + pos[v]
        except Exception as e:
            print(f'Skipping sidechain update because of the error:')
            print(e)

    if not as_numpy:
        pos = torch.from_numpy(pos.astype(np.float32))
        pos = pos.to(orig_device)

    return pos

def modify_sidechains(coords, edge_idx, subcomponentsMapping, subcomponents, torsion_updates):
    # iterate over all torsion updates and modify the corresponding atoms 
    for i, torsion_update in enumerate(torsion_updates):
        coords = modify_sidechain_torsion_angle(coords,
                                                edge_idx[i],
                                                subcomponentsMapping[i],
                                                subcomponents,
                                                torsion_update)
    return coords


def process_sc_rawdata(raw_data, new_mask=False):
    # pocket_info = raw_data['name']
    ligand_info = raw_data['resnums']
    complex_graph = raw_data['restypes']
    atoms = raw_data['atoms']
    atoms_coords = raw_data['coords']
    
    # iterate the atoms
    
    residue_info_dict = {}
    
    be_idx = 0
    end_idx = -1
    
    for i, atom in enumerate(atoms):
        pocket_resnum = ligand_info[i]
        pocket_restype = complex_graph[i]
        cur_res = f"{pocket_resnum}_{pocket_restype}"
        if cur_res in residue_info_dict.keys():
            residue_info_dict[cur_res][1] = i+1
        else:
            residue_info_dict[cur_res] = [i,i+1]
    
    # iterate
    res_nums = len(residue_info_dict)
    flexResidues = [{} for _ in range(res_nums)]
    flexResIdx = 0
    
    for i, atom in enumerate(atoms):
        pocket_resnum = ligand_info[i]
        pocket_restype = complex_graph[i]
        cur_res = f"{pocket_resnum}_{pocket_restype}"
        
        if cur_res in residue_info_dict:
            be_idx, end_idx = residue_info_dict[cur_res]
            
            atom_names = atoms[be_idx:end_idx]
            if new_mask:
                pass
                # flexResidues[flexResIdx] = get_sidechain_rotation_mask_2(atom_names, i, pocket_restype)
            else:
                flexResidues[flexResIdx] = get_sidechain_rotation_mask(atom_names, i)
            flexResIdx += 1
            # erase the residue info
            del residue_info_dict[cur_res]

    
    # move into tensors for now 
    subcomponentsMerged = []
    edge_idx_merged = []
    subcomponentsMapping = []
    res_n_bonds = []

    i = 0
    for res in flexResidues:
        subcomponents,edge_idxs = res["subcomponents"],res["edge_idx"]
        # save how many, i.e, which bonds and edges beling to which residue
        res_n_bonds.append(len(subcomponents))
        for component,edge_idx in zip(subcomponents,edge_idxs):
            subcomponentsMerged += component
            edge_idx_merged.append(edge_idx)
            subcomponentsMapping.append([i,i+len(component)])
            i += len(component)

    subcomponentsMerged = torch.tensor(subcomponentsMerged)
    subcomponentsMapping = torch.tensor(subcomponentsMapping)
    edge_idx_merged = torch.tensor(edge_idx_merged)
    residueNBondsMapping = torch.tensor(res_n_bonds)    
    atoms_coords = torch.tensor(np.vstack(atoms_coords))
    
    # flexible_info = {'subcomponents': subcomponentsMerged, 'subcomponentsMapping': subcomponentsMapping, 'edge_idx': edge_idx_merged, 'residueNBondsMapping': residueNBondsMapping}
    # return flexible_info
    
    
    # modify the original data
    sidechain_torsion_updates = np.random.uniform(0, 2*np.pi, size=len(edge_idx_merged))
    new_coords = modify_sidechains(atoms_coords, edge_idx_merged, subcomponentsMapping, subcomponentsMerged, sidechain_torsion_updates)
    
    return new_coords

            


