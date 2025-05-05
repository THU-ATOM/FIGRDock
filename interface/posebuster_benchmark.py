import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import os
from typing import Optional
from rdkit import Chem

def calculated_docking_grid_sdf(work_path, json_path, pdbid, ligid, add_size=10):
    input_path = os.path.join(work_path, f'{pdbid}_{ligid}.sdf')
    os.makedirs(json_path, exist_ok=True)
    output_grid = os.path.join(json_path, pdbid + '.json')
    add_size = add_size
    mol = Chem.MolFromMolFile(str(input_path), sanitize=False)
    coords = mol.GetConformer(0).GetPositions().astype(np.float32)
    min_xyz = [min(coord[i] for coord in coords) for i in range(3)]
    max_xyz = [max(coord[i] for coord in coords) for i in range(3)]
    center = np.mean(coords, axis=0)
    size = [abs(max_xyz[i] - min_xyz[i]) for i in range(3)]
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    size_x = size_x + add_size
    size_y = size_y + add_size
    size_z = size_z + add_size
    grid_info = {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "size_x": float(size_x),
        "size_y": float(size_y),
        "size_z": float(size_z)
    }
    with open(output_grid, 'w') as f:
        json.dump(grid_info, f, indent=4)
    print(pdbid)
    print('Center: ({:.6f}, {:.6f}, {:.6f})'.format(center_x, center_y, center_z))
    print('Size: ({:.6f}, {:.6f}, {:.6f})'.format(size_x, size_y, size_z))

import argparse
parser = argparse.ArgumentParser(description='Test on posebuster benchmark.\n \
    Metrices: ligand rmsd')
parser.add_argument(
    "--protein-path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/unimol_posebusters_eval_sets/posebusters/proteins",
    help='data path for input protein'
)
parser.add_argument(
    "--ligand-path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/unimol_posebusters_eval_sets/posebusters/ligands",
    help='data path for input ligand'
)
parser.add_argument(
    "--grid-path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/infer",
    help='data path for input grids'
)
parser.add_argument(
    "--output-sdf-path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/infer",
    help='data path for output sdf'
)
parser.add_argument(
    "--meta-info-file",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/unimol_posebusters_eval_sets/posebusters/posebuster_set_meta.csv",
    help='csv file path for protein/ligand id'
)
parser.add_argument(
    "--input-meta-info-file",
    type=str,
    default="posebuster_428_one2one.csv",
    help='csv file path for protein/ligand file path'
)
parser.add_argument(
    "--add-size",
    type=int,
    default=10,
    help='grid bound for ligand'
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help='numpy random seed'
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/data/protein/SKData/Uni-Mol-Docking/unimol_docking_v2_240517.pt", #12
    help='checkpoint path for inference model'
)
parser.add_argument(
    "--use-sidechain",
    type=str,
    default="rigid",
    choices=["rigid", "consider_sidechain", "perturb_sidechain"],
    help='define pocket scope'
)
parser.add_argument(
    "--task",
    type=str,
    default="debug",
    help='task name for predict sdf dir, define your task, may use model name'
)
args = parser.parse_args()
    

predict_sdf_dir = os.path.join(args.output_sdf_path, f'predict_sdf_posebuster428_grid{args.add_size}_{args.use_sidechain}_{args.task}')

# Step1: 计算grid files
protein_path = args.protein_path
ligand_path = args.ligand_path
meta_info_file = args.meta_info_file
add_size=args.add_size
grid_path = os.path.join(args.grid_path, f'posebuster428_grid{add_size}')
df = pd.read_csv(meta_info_file)
pdb_ids = list(df['pdb_code'].values)
lig_ids = list(df['lig_code'].values)
# generate docking grid json file
for pdbid, ligid in tqdm(zip(pdb_ids, lig_ids)):
    calculated_docking_grid_sdf(ligand_path, grid_path, pdbid, ligid, add_size=add_size)

# Step2: 将测试文件路径写入csv  
df = pd.DataFrame(columns=['input_protein', 'input_ligand', 'input_docking_grid', 'output_ligand_name'])
input_meta_info_file = args.input_meta_info_file
predict_name_suffix = 'predict'
for i, item in tqdm(enumerate(zip(pdb_ids,lig_ids))):
    pdbid, ligid = item
    single_protein_path = os.path.abspath(os.path.join(protein_path, pdbid + '.pdb'))
    single_ligand_path = os.path.abspath(os.path.join(ligand_path, f'{pdbid}_{ligid}.sdf'))
    single_grid_path = os.path.abspath(os.path.join(grid_path, pdbid + '.json'))
    predict_name = f'{pdbid}_{predict_name_suffix}'
    df.loc[i] = [single_protein_path, single_ligand_path, single_grid_path, predict_name]
print(df.info())
print(df.head(3))
df.to_csv(input_meta_info_file, index= False)

# Step 3: 调用demo.py 预测并后处理结果，保存在predict_sdf_dir路径下
# metrics
cmd = f"python demo.py --mode batch_one2one --batch-size 8 --steric-clash-fix --conf-size 10 --cluster \
        --input-batch-file {input_meta_info_file} \
        --output-ligand-dir {predict_sdf_dir} \
        --model-dir {args.checkpoint} \
        --use-sidechain {args.use_sidechain}" 
os.system(cmd)

def rmsd_func(holo_coords, predict_coords):
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        rmsd = np.sqrt(np.sum((predict_coords - holo_coords)**2) / sz[0])
        return rmsd
    return 1000.0

def rmsd_func_sym(holo_coords: np.ndarray, predict_coords: np.ndarray, mol: Optional[Chem.Mol] = None) -> float:
    """ Symmetric RMSD for molecules. """
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        if mol is not None:
            # get stereochem-unaware permutations: (P, N)
            base_perms = np.array(mol.GetSubstructMatches(mol, uniquify=False))
            # filter for valid stereochem only
            chem_order = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
            perms_mask = (chem_order[base_perms] == chem_order[None]).sum(-1) == mol.GetNumAtoms()
            base_perms = base_perms[perms_mask]
            noh_mask = np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()])
            # (N, 3), (N, 3) -> (P, N, 3), ((), N, 3) -> (P,) -> min((P,))
            best_rmsd = np.inf
            for perm in base_perms:
                rmsd = np.sqrt(np.sum((predict_coords[perm[noh_mask]] - holo_coords) ** 2) / sz[0])
                if rmsd < best_rmsd:
                    best_rmsd = rmsd

            rmsd = best_rmsd
        else:
            rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[0])
        return rmsd
    return 1000.0

def cal_rmsd_metrics(predict_dir, reference_dir, meta_info_file):
    df = pd.read_csv(meta_info_file)
    pdb_ids = list(df['pdb_code'].values)
    lig_ids = list(df['lig_code'].values)
    failed_num = 0
    rmsd_results, rmsd_sym_results = [], []
    for pdb_id, lig_id in tqdm(zip(pdb_ids, lig_ids)):
        target_ligand  = os.path.join(reference_dir, f'{pdb_id}_{lig_id}.sdf')
        predict_ligand = os.path.join(predict_dir, f'{pdb_id}_{predict_name_suffix}.sdf')
        target_supp = Chem.SDMolSupplier(target_ligand)
        target_mol = [mol for mol in target_supp if mol][0]
        target_mol = Chem.RemoveHs(target_mol)
        holo_coords = target_mol.GetConformer().GetPositions().astype(np.float32)
        target_atoms = [atom.GetSymbol() for atom in target_mol.GetAtoms()]
        predict_mol = Chem.MolFromMolFile(predict_ligand, sanitize=False)
        try:
            predict_coords = predict_mol.GetConformer().GetPositions().astype(np.float32)
        except:
            print(f'failed pdb_id: {pdb_id}')
            failed_num+=1
            continue
        predict_atoms = [atom.GetSymbol() for atom in predict_mol.GetAtoms()]
        if predict_atoms == target_atoms:
            rmsd = rmsd_func(holo_coords, predict_coords)
            rmsd_new = rmsd_func_sym(holo_coords, predict_coords, predict_mol)
            rmsd_results.append(rmsd)
            rmsd_sym_results.append(rmsd_new)
        else: 
            print(f'failed pdb_id: {pdb_id}')
            failed_num+=1
    rmsd_results = np.array(rmsd_results)
    rmsd_sym_results = np.array(rmsd_sym_results)
    return rmsd_results, rmsd_sym_results

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
    
    
rmsd_results, rmsd_sym_results = cal_rmsd_metrics(predict_dir = predict_sdf_dir, 
                                                  reference_dir=ligand_path, 
                                                  meta_info_file=meta_info_file)
print_result(rmsd_results)
print_result(rmsd_sym_results)