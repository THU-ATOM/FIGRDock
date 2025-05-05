import os
import argparse
import random
from rdkit import Chem
import numpy as np
from collections import defaultdict
import lmdb
import pickle
import re
import warnings
from Bio.PDB import PDBParser
biopython_parser = PDBParser()
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from predictor.processor import Processor
import json
from tqdm import tqdm
from multiprocessing import Pool
import torch
from flexible_docking_utils import process_sc_rawdata

parser = argparse.ArgumentParser(description='generate lmdb for training')
parser.add_argument(
    "--data_path",
    type=str,
    default="/data/protein/SKData/Uni-Mol-Docking/Docking_data/",
    help='data path for generating train and valid lmdb'
)
parser.add_argument(
    "--out_lmdb_dir", 
    type=str,
    default="/data/protein/BC_Data/Docking_Data/train_data/",
    help='lmdb output directory'
)
parser.add_argument(
    "--task_type", 
    type=str,
    default="sc_random_apo",
    choices=["sc", "sc_random_apo"],
    help='task type str for log file name'
)
parser.add_argument(
    "--example",
    action='store_true',
    help='if use small test set(size:200)'
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help='random seed'
)
parser.add_argument(
    "--mode", 
    type=str, 
    default="batch_one2one", 
    help="must not change"
)
parser.add_argument(
    "--nthreads", 
    type=int, 
    default=8, 
    help="num of threads for data preprocessing"
)
parser.add_argument(
    "--conf-size",
    default=10,
    type=int,
    help="number of conformers generated with each molecule",
)
parser.add_argument(
    "--cluster",
    action="store_true",
    help="whether preform conformer clustering when data preprocess",
)
parser.add_argument(
    "--use_current_ligand_conf", 
    action='store_true',
)
parser.add_argument(
    "--pocket-size",
    default=10,
    type=int,
    help="the radius of pocket context",
)

args = parser.parse_args()

example_str = "example" if args.example else "all"
out_lmdb_dir = os.path.join(args.output_lmdb_dir, f"protein_ligand_binding_pose_prediction_v2_{args.task_type}_{example_str}/")

# Step1: get train/valid split
moad_set = []
for entry in os.listdir(args.data_path):
    entry_path = os.path.join(args.data_path, entry)
    if os.path.isdir(entry_path):
        for complex_dir in os.listdir(entry_path):
            moad_set.append(entry + "/" + complex_dir)
random.seed(args.seed)  # 你可以使用任何数字作为种子
random.shuffle(moad_set)
if args.example:
    moad_set = moad_set[:200]
bound = len(moad_set) // 10
valid_set = moad_set[:bound]
train_set = moad_set[bound:]
def parse_pdb_from_path(path):
    ret = parse_pdb_structure_from_path(path)
    return ret[0]


def parse_pdb_structure_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure(os.path.basename(path), path)
        return structure



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

def check_res_intact(res_dict: dict) -> dict or None:
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
                    filtered_res_dict["resnums"].append(res_dict["resnums"][i])
                    filtered_res_dict["coords"].append(res_dict["coords"][i])

        if filtered_res_dict["atoms"] == intact_list:
            return filtered_res_dict
        else:
            return None
    else:
        return None

error_pocket_file_path = os.path.join(out_lmdb_dir, "fail_pocket.txt")
if os.path.exists(error_pocket_file_path):
    os.remove(error_pocket_file_path)
    
preprocessor = Processor.build_processors(
    args.mode, args.nthreads, conf_size=args.conf_size, cluster=args.cluster,
    use_current_ligand_conf=args.use_current_ligand_conf)
M = args.conf_size*10
N = args.conf_size

def parse_pocket_complex(content, radius=8):
    pdb_file, ligand_file = content[0], content[1]
    try:
        rec = parse_pdb_from_path(pdb_file)
    except:
        error_pocket_file = open(error_pocket_file_path,"a")
        error_pocket_file.write(pdb_file)
        return None
    
    retain_residue_list = []
    # read mol from sdf file
    ligand = Chem.SDMolSupplier(ligand_file)[0]
    smiles = Chem.MolToSmiles(ligand)
    ligand = Chem.AddHs(ligand)
    # get position of ligand
    ligand_pos = ligand.GetConformer().GetPositions().astype(np.float32)
    # get ligand atom type
    ligand_atoms_list = [atom.GetSymbol() for atom in ligand.GetAtoms()]
    ligand_center = np.mean(ligand_pos, axis=0)
    ligand_sphere_radius = np.sqrt(np.sum(np.square(ligand_pos - ligand_center), axis=1)).mean() # mean to max?
    
    holo_ligand = ligand
    mol_list = [ligand] * N
    try:
        coordinate_list = preprocessor.clustering_coords(ligand, M=M, N=N, seed=args.seed, cluster = args.cluster, removeHs=False, gen_mode='mmff') 
    except:
        try:
            coordinate_list = preprocessor.clustering_coords(ligand, M=M, N=N, seed=args.seed, cluster = args.cluster, removeHs=False, gen_mode='no_mmff') 
        except:
            print(f'Failed to generate conformers with RDKit: {ligand}, skipped!')  
            return None
    
    with open(ligand_file[:-4]+".json", "r") as file:
        box_dict = json.load(file)
    
    def _get_vertex(pocket: dict, axis: str) -> tuple:
            """
            Return the minimum and maximum values of the given axis

            Args:
            pocket (dict): pocket config
            axis (str): ["x", "y", "z"]

            Returns:
            A tuple of floats.
            """
            return (
                pocket["center_{}".format(axis)] \
                    - pocket["size_{}".format(axis)] / 2,
                pocket["center_{}".format(axis)] \
                    + pocket["size_{}".format(axis)] / 2
                )
    min_x, max_x = _get_vertex(box_dict, "x")
    min_y, max_y = _get_vertex(box_dict, "y")
    min_z, max_z = _get_vertex(box_dict, "z")

    residue_ids = []
    min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
    max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
    for i, chain in enumerate(rec):
        for res_idx, residue in enumerate(chain):
            for atom in residue:
                _rescoor = np.array(list(atom.get_vector())).reshape(-1,3)
                mapping = (_rescoor > min_array) & (_rescoor < max_array)
                if (mapping.sum(-1) == 3).sum() > 0:
                    residue_ids.append(f'{chain.get_id()}_{residue.get_id()[1]}{residue.get_id()[2]}'.strip())
                    break
    res_info_dict = {resnum: defaultdict(list) for resnum in residue_ids}
    
    for all_atom in rec.get_atoms():
        residu_id = f'{all_atom.get_parent().get_parent().get_id()}_{all_atom.get_parent().get_id()[1]}{all_atom.get_parent().get_id()[2]}'.strip()
        if residu_id in res_info_dict:
            pocket_atom = all_atom.get_name()
            pocket_restype = all_atom.get_parent().get_resname()
            coordinates = list(all_atom.get_coord())
            res_info_dict[residu_id]["restypes"].append(pocket_restype)
            res_info_dict[residu_id]["atoms"].append(pocket_atom)
            res_info_dict[residu_id]["coords"].append(coordinates)
            res_info_dict[residu_id]["resnums"].append(residu_id)
    
    # for resnum, info in res_info_dict.items():
    #     res_cord_array = np.array(info["coords"]).astype(np.float32)
    #     min_dist = np.min(np.linalg.norm(ligand_pos[:, None] - res_cord_array[None, :], axis=-1))
    #     if min_dist <= ligand_sphere_radius + radius:
    #         retain_residue_list.append(resnum)
    
    
    # check side chain intact
    pocket_info_dict = {
        "name": f"{os.path.basename(ligand_file.split('.')[0])}",
        "resnums": [],
        "restypes": [],
        "atoms": [],
        "coords": []
    }

    for resnum in residue_ids:
        filtered_res_dict = check_res_intact(res_info_dict[resnum])
        if filtered_res_dict is None:
            print(f'check pocket {pdb_file}, residue {resnum} failed')
            # return None
        else:
            pocket_info_dict["resnums"].extend(filtered_res_dict["resnums"])
            pocket_info_dict["restypes"].extend(filtered_res_dict["restypes"])
            pocket_info_dict["atoms"].extend(filtered_res_dict["atoms"])
            pocket_info_dict["coords"].extend(filtered_res_dict["coords"])

    pocket_info_dict["coords"] = [np.array(pocket_info_dict["coords"])]
    
    # Less atoms in this setting is due to not considering Hydrogen which is removed in training.
    if args.task_type == "sc_random_apo":
        pocket_coordinates = process_sc_rawdata(pocket_info_dict, new_mask=False)
        pocket_coordinates = [np.array(pocket_coordinates)]
    else:
        pocket_coordinates = pocket_info_dict["coords"]
    
    data = {
        "atoms": ligand_atoms_list,
        "coordinates": coordinate_list,
        "mol_list": mol_list,
        "pocket_atoms": pocket_info_dict["atoms"], 
        "pocket_coordinates": pocket_coordinates,
        # "side": side,
        "residue": pocket_info_dict["resnums"],
        "config": box_dict,
        "holo_coordinates": [ligand_pos],
        "holo_mol": holo_ligand,
        "holo_pocket_coordinates": pocket_info_dict["coords"],
        "smi": smiles,
        "pocket": pdb_file,
    }
    return pickle.dumps(
        data, protocol=-1,
        )
    # except Exception as e:
    #     print(f'parse pocket {pdb_file} failed, ligand_file is {ligand_file}, and error is {e}')
    #     return None



def write_lmdb_(complex_dir_list, outputfilename):
    pdb_file_list = []
    ligand_file_list = []
    
    for complex_dir in tqdm(complex_dir_list):
        complex_name = complex_dir.split("/")[-1]
        intput_protein = os.path.join(complex_dir, "protein.pdb")
        ligand_name_list = []
        for _, _, files in os.walk(complex_dir):
            for name in files:
                if name.split(".")[1] == "sdf":
                    ligand_name = name.split(".")[0]
                    ligand_name_list.append(ligand_name)
        for ligand_name in ligand_name_list:
            input_ligand = os.path.join(complex_dir, ligand_name+".sdf")
            ligand_file_list.append(input_ligand)
            pdb_file_list.append(intput_protein)
        
    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn_write = env_new.begin(write=True)
    print("Start preprocessing data...")
    print(f'Number of ligands: {len(ligand_file_list)}')
    seed = [args.seed] * len(input_ligand)
    content_list = zip(pdb_file_list, ligand_file_list)
    with Pool(args.nthreads) as pool:
        i = 0
        failed_num = 0
        for inner_output in tqdm(pool.imap(parse_pocket_complex, content_list)):
            if inner_output is not None:
                txn_write.put(f"{i}".encode("ascii"), inner_output)
                i+=1
            elif inner_output is None: 
                failed_num += 1
        txn_write.commit()
        env_new.close()
    print(f'Total num: {len(ligand_file_list)}, Success: {i}, Failed: {failed_num}')
    print("Done!")
            
        
for data_set, lmdb_name in [(train_set, "train"), (valid_set, "valid")]:
    complex_dir_list = [os.path.join(args.data_path, _) for _ in data_set]
    if not os.path.exists(out_lmdb_dir):
        os.mkdir(out_lmdb_dir)
    outputfilename = os.path.join(out_lmdb_dir, f'{lmdb_name}.lmdb')
    write_lmdb_(complex_dir_list, outputfilename)