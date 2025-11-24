# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset

from . import flexible_docking_utils

from . import data_utils
from rdkit import Chem

class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        residue,
        is_train=True,
        conf_size=10,
        use_flexible_docking=False,
        pocket_dict_len=9,
        sc_aug=True,
        add_feature=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.residue = residue
        self.is_train = is_train
        self.conf_size = conf_size
        self.use_flexible_docking = use_flexible_docking
        self.pocket_dict_len = pocket_dict_len
        self.sc_aug = sc_aug
        self.add_feature = add_feature
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        if self.pocket_dict_len != 10 or self.use_flexible_docking in ["base_flex_sc", "base_flex_all"]:
            pocket_atoms = np.array(
                [item for item in self.dataset[smi_idx][self.pocket_atoms]]
            )
        else:
            pocket_atoms = np.array(
                [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
            )
        if self.use_flexible_docking in ["flex_sc", "base_flex_sc"]:
            masked_tokens = np.array([(item=="C") or (item=="CA") or (item=="N") or (item=="O") 
                                    for item in self.dataset[smi_idx][self.pocket_atoms]]) 
        else:
            masked_tokens = np.array([0 for _ in self.dataset[smi_idx][self.pocket_atoms]])
        if self.use_flexible_docking == "rigid":
            pocket_coordinates = np.array(self.dataset[smi_idx][self.holo_pocket_coordinates][0])
        else:
            pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
        if self.is_train:
            holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
            holo_pocket_coordinates = np.array(
                self.dataset[smi_idx][self.holo_pocket_coordinates][0]
            )
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[smi_idx]["smi"]
        pocket = self.dataset[smi_idx]["pocket"]
        # id = self.dataset[smi_idx]['id']
        
        
        # for the las constraint
        holo_mol = self.dataset[smi_idx]["holo_mol"]
        
        atoms1 = [atom.GetSymbol() for atom in holo_mol.GetAtoms()]
        # remove H atoms
        # holo_mol_remove_h = Chem.RemoveHs(holo_mol) # TODO, the order may different from the original mol after remove Hs
        
        # get torsion angle idx
        rw_mol = Chem.RWMol(holo_mol)
        remove_h_idx = [i for i, atom in enumerate(atoms1) if atom == "H"]
        
        remove_h_idx.sort(reverse=True)

        for idx in remove_h_idx:
            rw_mol.RemoveAtom(idx)
        
        holo_mol_remove_h = rw_mol.GetMol()
        
        atoms2 = [atom.GetSymbol() for atom in holo_mol_remove_h.GetAtoms()]
        
        ligand_feats = data_utils.extract_torchdrug_feature_from_mol(holo_mol_remove_h, has_LAS_mask=True)
        compound_LAS_edge_index = ligand_feats # todo using the rdkit init coords
        compound_LAS_edge_index = compound_LAS_edge_index.T
        atoms_with_h = []
        atoms_remove_h = [ele for ele in atoms if ele != "H"]
        assert len(atoms_remove_h) == len(atoms2), f"data error: {len(atoms_remove_h)} vs {len(atoms2)} details: {atoms_remove_h} vs {atoms2}"
        if compound_LAS_edge_index.max() >= len(atoms2):
            print('data error')
        
        
        if self.use_flexible_docking != "rigid":
            residue = self.dataset[smi_idx][self.residue]
            raw_data = {
                "resnums": residue,
                "restypes": residue,
                "atoms": pocket_atoms,
                "coords": pocket_coordinates
            }
            if self.sc_aug:
                pocket_coordinates = flexible_docking_utils.process_sc_rawdata(raw_data)
            pocket_coordinates = np.array(pocket_coordinates)

        if self.use_flexible_docking in ["base_flex_sc", "base_flex_all"]:
            pocket_atoms = np.array(
                [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
            )
        ret =  {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
            "masked_tokens": masked_tokens,
            "compound_LAS_edge_index": compound_LAS_edge_index,
            # 'id': id,
        }
        if self.add_feature:
            for feature in ["atom_features_list", "bond_type_matrix", "restype_tokens"]:
                ret[feature] = torch.tensor(np.array(self.dataset[smi_idx][feature])).long()
            ret["lm_embeddings"] = torch.tensor(np.array(self.dataset[smi_idx]["lm_embeddings"]), dtype=torch.float16)
        return ret

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
