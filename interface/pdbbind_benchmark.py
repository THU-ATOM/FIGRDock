import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import os
from typing import Optional
from rdkit import Chem
from predictor.processor import Processor, sdf_or_mol, read_molecule
# from posebuster_benchmark import rmsd_func, rmsd_func_sym, print_result
import time
import lmdb
import pickle
from pdbbind_benchmark_sc_utils import get_predict_pdb, cal_pocket_rmsd_metrics
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Test on posebuster benchmark.\n \
    Metrices: ligand rmsd')
parser.add_argument(
    "--data_path",
    type=str,
    default="/data/protein/SKData/DiffDock-Pocket/data/PDBBIND_atomCorrected",
    help='data path for generating train and valid lmdb'
)
# /data/protein/SKData/DiffDock-Pocket/data/PDBBIND_atomCorrected/ for 19
# /mnt/nfs-ssd/data/fengshikun/DiffPocket/data/PDBBIND_atomCorrected/ for drug machine

parser.add_argument(
    "--test-lmdb",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/pdbbind/unimol_bindnet_8A_all_with_Atom/test.lmdb",
    help='data path for test lmdb'
)
parser.add_argument(
    "--output-sdf-path",
    type=str,
    default="./infer_pdbbind",
    help='data path for output sdf and pdb'
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
)
parser.add_argument(
    "--conf-size",
    type=int,
    default=10,
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help='numpy random seed'
)
parser.add_argument(
    "--device",
    type=int,
    default=5,
    help='cuda device'
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/data/protein/SKData/Uni-Mol-Docking/unimol_docking_v2_240517.pt", #12
    help='checkpoint path for inference model'
)
parser.add_argument(
    "--use-flexible-docking",
    default="flex_sc",
    choices=["rigid", "flex_sc", "flex_all", "base_flex_sc", "base_flex_all"],
    type=str,
    help="if use half size dataset",
)
parser.add_argument(
    "--pocket-dict",
    default="dict_sidechain.txt",
    choices=["dict_sidechain.txt", "dict_pkt.txt"],
    type=str,
    help="if use half size dataset",
)
parser.add_argument(
    "--max-pocket-atoms",
    type=int,
    default=256,
    help='cuda device'
)
parser.add_argument(
    "--sc_aug",
    type=int,
    default=1,
    help="do the sidechain augmentation",
)
parser.add_argument(
    "--task",
    type=str,
    default="debug",
    help='task name for predict sdf dir, define your task, may use model name'
)
parser.add_argument(
    "--no-clash-fix",
    action='store_true',
    help='if no clash fixing is required'
)
parser.add_argument(
    "--feat_position",
    type=str,
    default="None",
    choices=["None", "front", "back"],
    help="recycling nums of decoder",
)

parser.add_argument(
    "--coord_decode_total_iter",
    type=int,
    default=4,
    help="number of stack layers",
)
        
parser.add_argument(
    "--geom_reg_steps",
    type=int,
    default=0,
    help="LAS constraint steps",
)

# parser add the layer number of each module
parser.add_argument(
    "--coord_decode_layers",
    type=int,
    default=4,
    help="number of encoder layers",
)
parser.add_argument(
    "--las_init_opti",
    type=int,
    default=0,
    help="LAS optimization at the gradient of init coords in dock_with_gradient",
)


parser.add_argument(
    "--split_encoder",
    type=int,
    default=0,
    help="the split encoder for the mol and pocket",
)


args = parser.parse_args()
# ensure checkpoint exist or download for google drive
if not os.path.exists(args.checkpoint):
    checkpointdir = os.path.dirname(args.checkpoint)
    print(f"⚠ Checkpoint not exist: {args.checkpoint}", flush=True)
    print(f"Downloading from Google Drive...", flush=True)
    import gdown, torch, pathlib as P
    GDRIVE_FILE_ID = "1YNJ-HgasDS8bRDMAMKgYS0cs7H4-BP8R"
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    os.makedirs(checkpointdir, exist_ok=True)
    gdown.download(url, args.checkpoint, quiet=False)

assert os.path.exists(args.data_path), f'data path not exists: {args.data_path}'
assert os.path.exists(args.checkpoint), f'checkpoint not exists: {args.checkpoint}'
assert os.path.exists(args.output_sdf_path), f'output sdf path not exists: {args.output_sdf_path}'
predict_sdf_dir = os.path.join(args.output_sdf_path, f'predict_sdf_pdbbind_radius8_{args.use_flexible_docking}_{args.task}')
if not os.path.exists(predict_sdf_dir):
    os.mkdir(predict_sdf_dir)

if args.test_lmdb.split("/")[-1] == "test_holo.lmdb":
    suffix = "protein_processed_fix"
elif args.test_lmdb.split("/")[-1] == "test_apo.lmdb":
    suffix = "protein_esmfold_aligned_tr_fix"
else:
    assert NotImplementedError
output_ligand_name = []
input_protein = []
input_ligand = []
env = lmdb.open(
    args.test_lmdb,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)
with env.begin() as txn:
    _keys = list(txn.cursor().iternext(values=False))
for idx in range(len(_keys)):
    datapoint_pickled = env.begin().get(f"{idx}".encode("ascii"))
    data = pickle.loads(datapoint_pickled)
    ligand_name = data["pocket"].split("/")[-2]
    ligand_file = sdf_or_mol(os.path.join(os.path.join(args.data_path, ligand_name), f"{ligand_name}_ligand"))
    if ligand_file == None:
        print("bad case for: ligand", ligand_name)
        continue
    
    output_ligand_name.append(ligand_name)
    input_protein.append(os.path.join(os.path.join(args.data_path, ligand_name), f"{ligand_name}_{suffix}.pdb"))
    input_ligand.append(ligand_file)


print("Test samples counts: ", len(input_ligand))

lmdb_name = os.path.basename(args.test_lmdb).split(".")[0]
# work on pdbbind lmdb
user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figrdock")
script_path = os.path.join(user_dir, "infer.py")
assert os.path.exists(script_path), f'script path not exists: {script_path}'
if args.use_flexible_docking in ["base_flex_sc", "base_flex_all"]:
    assert args.pocket_dict == "dict_pkt.txt"
loss = "flexible_docking_pose_v2" if args.use_flexible_docking != "rigid" else "docking_pose_v2"
feature_args = f"--add_feature 1 --feat_position {args.feat_position}" if args.feat_position != "None" else ""
cmd = f'\
    cp {os.path.join(os.path.dirname(args.test_lmdb), "dict_mol.txt") } {os.path.abspath(predict_sdf_dir)} \n\
    cp {os.path.join(os.path.dirname(args.test_lmdb), args.pocket_dict) } {os.path.abspath(predict_sdf_dir)} \n\
    cp {args.test_lmdb} {os.path.abspath(predict_sdf_dir)} \n\
    CUDA_VISIBLE_DEVICES={args.device} /opt/conda/bin/python {script_path} --user-dir {user_dir} {os.path.abspath(predict_sdf_dir)} --valid-subset {lmdb_name} \
        --results-path {os.path.abspath(predict_sdf_dir)} \
        --num-workers 8 --ddp-backend=c10d --batch-size {args.batch_size} \
        --task docking_pose_v2 --loss {loss} --arch docking_pose_v2 \
        --conf-size {args.conf_size} \
        --dist-threshold 8.0 --recycling 4 \
        --path {args.checkpoint}  \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --pocket-dict {args.pocket_dict} \
        --use-flexible-docking {args.use_flexible_docking} \
        --sc_aug {args.sc_aug} \
        --max-pocket-atoms {args.max_pocket_atoms} \
        --geom_reg_steps {args.geom_reg_steps} \
        --coord_decode_total_iter {args.coord_decode_total_iter} \
        --coord_decode_layers {args.coord_decode_layers} \
        --las_init_opti {args.las_init_opti} \
        --split_encoder {args.split_encoder} \
         {feature_args}'
os.system(cmd)


# return results path and lmdb path
lmdb_name = os.path.basename(args.test_lmdb).split(".")[0]
pkl_file = os.path.join(os.path.abspath(predict_sdf_dir), f'{lmdb_name}.pkl')
lmdb_file = os.path.join(os.path.abspath(predict_sdf_dir), f'{lmdb_name}.lmdb')
    
# save ligand and protein
start_time = time.time()
postprocessor = Processor.build_processors("batch_one2one", conf_size=args.conf_size)
mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = postprocessor.postprocess_data_pre(pkl_file, lmdb_file)
output_ligand_sdf = postprocessor.get_sdf(mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, os.path.abspath(predict_sdf_dir), tta_times=args.conf_size)      

if suffix == "protein_esmfold_aligned_tr_fix": # if apo
    pocket_dict = os.path.join(predict_sdf_dir, args.pocket_dict)
    input_protein, target_protein, predict_pocket_atom_dict =  get_predict_pdb(lmdb_file, pkl_file, args.batch_size, args.conf_size, predict_sdf_dir, args.max_pocket_atoms, pocket_dict, data_path=args.data_path)
    print("input_protein: ", input_protein)

# optimize
if not args.no_clash_fix:
    output_ligand_sdf = postprocessor.clash_fix(output_ligand_sdf, input_protein, input_ligand)

print('output ligands path:\n', output_ligand_sdf)
end_time = time.time()
execution_time = end_time - start_time
print("Average time: ", execution_time/len(input_ligand), "sec.")
print("Total time: ", execution_time, "sec.")
print('All processes done!')

# calculate metrices both ligand and protein
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
            if len(base_perms) == 0: return 1000.0
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

def cal_rmsd_metrics(predict_dir):
    failed_num = 0
    rmsd_results, rmsd_sym_results = [], []
    csv_result = []
    for i, lig_id in tqdm(enumerate(output_ligand_name)):
        target_ligand  = input_ligand[i]
        predict_ligand = os.path.join(predict_dir, f'{lig_id}.sdf')
        if not os.path.exists(predict_ligand):
            failed_num += 1
            pass
        target_mol = read_molecule(target_ligand)
        target_mol = Chem.RemoveHs(target_mol)
        holo_coords = target_mol.GetConformer().GetPositions().astype(np.float32)
        target_atoms = [atom.GetSymbol() for atom in target_mol.GetAtoms()]
        target_noh_mask = [a != 'H' for a in target_atoms]
        holo_coords = holo_coords[target_noh_mask]
        target_atoms = [a for a in target_atoms if a != 'H']
        predict_mol = Chem.MolFromMolFile(predict_ligand, sanitize=False)
        try:
            predict_coords = predict_mol.GetConformer().GetPositions().astype(np.float32)
        except:
            print(f'failed {lig_id} to predict coords')
            failed_num+=1
            continue
        predict_atoms = [atom.GetSymbol() for atom in predict_mol.GetAtoms()]
        predict_noh_mask = [a != 'H' for a in predict_atoms]
        predict_coords = predict_coords[predict_noh_mask]
        predict_atoms = [a for a in predict_atoms if a != 'H']
        if predict_atoms == target_atoms:
            rmsd = rmsd_func(holo_coords, predict_coords)
            rmsd_new = rmsd_func_sym(holo_coords, predict_coords, predict_mol)
            if rmsd >10 or rmsd_new >10:
                print("rmsd too large for ", lig_id, rmsd, rmsd_new)
            rmsd_results.append(rmsd)
            rmsd_sym_results.append(rmsd_new)
            csv_result.append([lig_id, rmsd_new, len(target_atoms)])
        else: 
            print(f'failed: {lig_id} {predict_atoms} {target_atoms}')
            failed_num+=1
    rmsd_results = np.array(rmsd_results)
    rmsd_sym_results = np.array(rmsd_sym_results)
    return rmsd_results, rmsd_sym_results, csv_result

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

rmsd_results, rmsd_sym_results, lig_csv_result = cal_rmsd_metrics(predict_dir = predict_sdf_dir)
print_result(rmsd_results)
print_result(rmsd_sym_results)
lig_csv_result = [["complex_name", "lig_rmsd", "lig_atom"]] + lig_csv_result
# calculate pocket rmsd

if suffix == "protein_esmfold_aligned_tr_fix": # if apo
    pocket_rmsd_results, pocket_sym_rmsd_results, pocket_csv_result = cal_pocket_rmsd_metrics(input_protein, target_protein, predict_pocket_atom_dict)
    print_result(pocket_rmsd_results)
    print_result(pocket_sym_rmsd_results)

    for idx, (complex1, lig_rmsd, _) in enumerate(lig_csv_result[1:]):
        for complex2, pocket_rmsd in pocket_csv_result:
            if complex1 == complex2:
                lig_csv_result[idx+1][2] = (pocket_rmsd)
                break
    lig_csv_result[0][2] = "pocket_rmsd"

