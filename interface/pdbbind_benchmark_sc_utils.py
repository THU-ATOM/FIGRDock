import lmdb 
import pickle
from tqdm import tqdm
import os
import copy
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from rdkit.Chem.rdchem import BondType as BT

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

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
        'misc',
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True, 'misc'],
    'possible_is_in_ring3_list': [False, True, 'misc'],
    'possible_is_in_ring4_list': [False, True, 'misc'],
    'possible_is_in_ring5_list': [False, True, 'misc'],
    'possible_is_in_ring6_list': [False, True, 'misc'],
    'possible_is_in_ring7_list': [False, True, 'misc'],
    'possible_is_in_ring8_list': [False, True, 'misc'],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
    'possible_flexible_sidechains': {'ARG','HIS','LYS','ASP','GLU','SER','THR','ASN','GLN','CYS','SEC','GLY','PRO','ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP'}
}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0) 

bond_type_to_value = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3
}

def order_atoms_in_residue(res, atom):
    """
    An order function that sorts atoms of a residue.
    Atoms N, CA, C, O always come first, thereafter the rest of the atoms are sorted according
    to how they appear in the chemical components. Hydrogens come last and are not sorted.
    """
    if atom.name == "OXT":
        return 999
    elif atom.element == "H":
        return 1000
    if res.resname in SORTING_DICT:
        if atom.name in SORTING_DICT[res.resname]:
            return SORTING_DICT[res.resname].index(atom.name)
    else:
        raise Exception("Unknown residue", res.resname)
    raise Exception(f"Could not find atom {atom.name} in {res.resname}")

def _sort_atoms_by_element(_protein):
        for res in _protein.get_residues():
            res.child_list.sort(key=lambda atom: order_atoms_in_residue(res, atom))

def _remove_hs(_protein):
    for res in _protein.get_residues():
        atoms_to_remove = []
        for atom in res:
            if atom.element == 'H':
                atoms_to_remove.append(atom)
        for atom in atoms_to_remove:
            res.detach_child(atom.id)

def get_predict_pdb(test_lmdb, test_pickle, batch_size, conf_size, output_dir, max_pocket_atoms, pocket_dict, data_path=None, df=None):
    env = lmdb.open(
        test_lmdb,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    output_protein_list = []
    target_protein_list = []
    predict_pocket_atom_dict = []

    with env.begin() as txn:
        _keys = list(txn.cursor().iternext(values=False))
    with open(test_pickle, 'rb') as file:  # 替换 'your_file.pkl' 为你的文件名
        test_data = pickle.load(file)
    with open(pocket_dict) as file:
        pocket_dictionary_list = [line.strip() for line in file.readlines()]

    for idx in tqdm(range(len(_keys))):
        datapoint_pickled = env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        if df is not None:
            complex_name = data["pocket"].split("/")[-1].split("_")[1]
            data_info = df[df['pdb'] == complex_name]
            apo_protein_path = os.path.join(data_path, data_info['af2File'].item()[1:])
        elif data_path is not None:
            complex_name = data["pocket"].split("/")[-1].split("_")[0]
            apo_protein_path = os.path.join(data_path, complex_name, f"{complex_name}_protein_esmfold_aligned_tr_fix.pdb")
        else:
            apo_protein_path = data["pocket"]
        pocket_atoms = data["pocket_atoms"]
        residue_list = data["residue"]
        pocket_center = data["pocket_coordinates"][0].mean(axis=0)
        parser = PDBParser()
        structure = parser.get_structure("structure", apo_protein_path)
        _remove_hs(structure)
        _sort_atoms_by_element(structure)
        if df is not None:
            holo_protein_path = os.path.join(data_path, data_info['pdbFile'].item()[1:])
        else:
            holo_protein_path = os.path.join(os.path.dirname(apo_protein_path), f"{complex_name}_protein_processed_fix.pdb")
        target_protein_list.append(holo_protein_path)
        predict_pocket_atom = {}

        d = (idx * conf_size) // batch_size
        r = (idx * conf_size) % batch_size
        min_pocket_prmsd = 1e9
        for _ in range(conf_size):
            if test_data[d]["pocket_prmsd_score"][r] < min_pocket_prmsd:
                min_pocket_prmsd = test_data[d]["pocket_prmsd_score"][r]
                best_idx = [d,r]
            if r == batch_size-1:
                d += 1
                r = 0
            else:
                r += 1
        pocket_coord_predict = test_data[best_idx[0]]["pocket_coord_predict"][best_idx[1]][1:-1]
        if len(pocket_atoms) > max_pocket_atoms:
            open(os.path.join(output_dir, "large.log"), "a").write(apo_protein_path+"\n")
            cropped_pocket_atoms = test_data[best_idx[0]]["pocket_atoms"][best_idx[1]][1:-1]
            c_idx = 0
            new_pocket_atoms = []
            new_residue_list = []
            new_pocket_coordinate = []
            for patom, pres, pcoord in zip(pocket_atoms, residue_list, data["pocket_coordinates"][0]):
                if pocket_dictionary_list[cropped_pocket_atoms[c_idx]] == patom:
                    c_idx += 1
                    new_pocket_atoms.append(patom)
                    new_residue_list.append(pres)
                    new_pocket_coordinate.append(pcoord)
                if c_idx >= max_pocket_atoms: break
            assert len(new_pocket_atoms) == len(cropped_pocket_atoms), f"{len(new_pocket_atoms)} {len(cropped_pocket_atoms)}"
            assert len(new_residue_list) == len(new_pocket_coordinate) == len(cropped_pocket_atoms)
            pocket_atoms = new_pocket_atoms
            residue_list = new_residue_list
            pocket_center = np.array(new_pocket_coordinate).mean(axis=0)
        
        cnt = 0
        for model in structure:
            for chain in model:
                for res in chain:
                    residue_id = f'{res.get_parent().get_id()}_{res.get_id()[1]}{res.get_id()[2]}'.strip()
                    if residue_id in residue_list:
                        start = residue_list.index(residue_id)
                        end = start + residue_list.count(residue_id)
                        res_atom_list = pocket_atoms[start:end]
                        for atom in res:
                            if atom.get_name() in res_atom_list:
                                atom.coord = pocket_coord_predict[start + res_atom_list.index(atom.get_name())].cpu() + pocket_center
                                predict_pocket_atom[f"{residue_id}-{atom.get_name()}"] = cnt
                                cnt += 1
        assert cnt == len(pocket_atoms)
        predict_pocket_atom["length"] = cnt
        predict_pocket_atom_dict.append(predict_pocket_atom)

        io = PDBIO()
        io.set_structure(structure)
        pdb_output_path = os.path.join(output_dir, f"{complex_name}_predict.pdb")
        io.save(pdb_output_path)
        output_protein_list.append(pdb_output_path)

    return output_protein_list, target_protein_list, predict_pocket_atom_dict


def cal_pocket_rmsd_metrics(input_protein, target_protein, predict_pocket_atom_dict):
    pocket_rmsd_results = []
    pocket_sym_rmsd_results = []
    pocket_csv_result = []
    for predict_pt, target_pt, pocket_atom in zip(input_protein, target_protein, predict_pocket_atom_dict):
        parser = PDBParser()
        predict_structure = parser.get_structure("structure", predict_pt)
        target_structure = parser.get_structure("structure", target_pt)
        _remove_hs(predict_structure)
        _sort_atoms_by_element(predict_structure)
        _remove_hs(target_structure)
        _sort_atoms_by_element(target_structure)
        
        predict_atom_coord_list = [[-1e9, -1e9,-1e9]] * pocket_atom["length"]
        target_atom_coord_list = [[-1e9, -1e9,-1e9]] * pocket_atom["length"]
        for p_atom, t_atom in zip(predict_structure.get_atoms(), target_structure.get_atoms()):
            res = p_atom.get_parent()
            residue_id = f'{res.get_parent().get_id()}_{res.get_id()[1]}{res.get_id()[2]}'.strip()
            if f"{residue_id}-{p_atom.get_name()}" in pocket_atom.keys():
                index = pocket_atom[f"{residue_id}-{p_atom.get_name()}"]
                predict_atom_coord_list[index] = p_atom.coord
                target_atom_coord_list[index] = t_atom.coord
        predict_atom_coord_list = np.array(predict_atom_coord_list)
        target_atom_coord_list = np.array(target_atom_coord_list)

        assert not any(np.array_equal(coord, [-1e9, -1e9,-1e9]) for coord in predict_atom_coord_list)
        assert not any(np.array_equal(coord, [-1e9, -1e9,-1e9]) for coord in target_atom_coord_list)

        rmsd = np.sqrt(np.sum((predict_atom_coord_list - target_atom_coord_list) ** 2) / len(predict_atom_coord_list))
        pocket_rmsd_results.append(rmsd)

        # cal sym rmsd
        for p_atom, t_atom in zip(predict_structure.get_atoms(), target_structure.get_atoms()):
            res = p_atom.get_parent()
            residue_id = f'{res.get_parent().get_id()}_{res.get_id()[1]}{res.get_id()[2]}'.strip()
            restype = res.get_resname()
            index1 = index2 = -1
            p_atom_name = p_atom.get_name()
            if restype == "LEU" and p_atom_name == "CD1" or \
                restype == "PHE" and p_atom_name == "CD1" or \
                restype == "PHE" and p_atom_name == "CE1" or \
                restype == "TYR" and p_atom_name == "CD1" or \
                restype == "TYR" and p_atom_name == "CE1" or \
                restype == "VAL" and p_atom_name == "CG1":
                if f"{residue_id}-{p_atom_name}" in pocket_atom:
                    index1 = pocket_atom[f"{residue_id}-{p_atom_name}"]
                if f"{residue_id}-{p_atom_name[:-1]}2" in pocket_atom:
                    index2 = pocket_atom[f"{residue_id}-{p_atom_name[:-1]}2"]
            if index1 != -1 and index2 != -1:
                origin = np.array([predict_atom_coord_list[index1], predict_atom_coord_list[index2]])
                symmetry = np.array([predict_atom_coord_list[index2], predict_atom_coord_list[index1]])
                target = np.array([target_atom_coord_list[index1], target_atom_coord_list[index2]])
                if np.sum((symmetry - target) ** 2) < np.sum((origin - target) ** 2):
                    predict_atom_coord_list[index1], predict_atom_coord_list[index2] = symmetry
        
        rmsd = np.sqrt(np.sum((predict_atom_coord_list - target_atom_coord_list) ** 2) / len(predict_atom_coord_list))
        pocket_sym_rmsd_results.append(rmsd)
        complex_name =  predict_pt.split("/")[-1].split("_")[0]
        pocket_csv_result.append([complex_name, rmsd])

    pocket_rmsd_results = np.array(pocket_rmsd_results)
    pocket_sym_rmsd_results = np.array(pocket_sym_rmsd_results)
    return pocket_rmsd_results, pocket_sym_rmsd_results, pocket_csv_result

if __name__ == '__main__':

    test_lmdb = "/data/protein/BC_Data/Docking_Data/pdbbind/unimol_bindnet_8A_all_with_Atom_test_good_data/test_apo.lmdb"
    test_pickle = "/data/protein/BC_Data/Docking_Data/infer_pdbbind_flex/predict_sdf_pdbbind_radius8_flex_all_PDBBind+apo+pretrain11last-norm+8A+510+flexall+gooddata+pdbfix/test_apo.pkl"
    batch_size = 8
    conf_size = 10
    output_dir = "/data/protein/BC_Data/tmp/modified_structure_debug/"
    max_pocket_atoms = 510
    pocket_dict = "/data/protein/BC_Data/Docking_Data/pdbbind/unimol_bindnet_8A_all_with_Atom_test_good_data/dict_sidechain.txt"

    input_protein, target_protein, predict_pocket_atom_dict = get_predict_pdb(test_lmdb, test_pickle, batch_size, conf_size, output_dir, max_pocket_atoms, pocket_dict)

    pocket_rmsd_results, pocket_sym_rmsd_results = cal_pocket_rmsd_metrics(input_protein, target_protein, predict_pocket_atom_dict)

    def print_result(rmsd_results):
        print('*'*100)
        print(f'results length: {len(rmsd_results)}')
        print('RMSD < 0.5 : ', np.mean(rmsd_results<0.5))
        print('RMSD < 1.0 : ', np.mean(rmsd_results<1.0))
        print('RMSD < 1.5 : ', np.mean(rmsd_results<1.5))
        print('RMSD < 2.0 : ', np.mean(rmsd_results<2.0))
        print('RMSD < 3.0 : ', np.mean(rmsd_results<3.0))
        print('RMSD < 5.0 : ', np.mean(rmsd_results<5.0))
        print('avg RMSD : ', np.mean(rmsd_results))
        print('median RMSD : ', np.median(rmsd_results))

    print_result(pocket_rmsd_results)
    print_result(pocket_sym_rmsd_results)