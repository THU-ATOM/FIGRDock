import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import os
from typing import Optional
from rdkit import Chem
from predictor.processor import Processor, sdf_or_mol
import time
import lmdb
import pickle
from pdbbind_benchmark_sc_utils import get_general_predict_pdb
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Test on posebuster benchmark.\n \
    Metrices: ligand rmsd')
parser.add_argument(
    "--data-path",
    type=str,
    default="/data/protein/SKData/DiffDock-Pocket/data/PDBBIND_atomCorrected",
    help='data path for generating train and valid lmdb'
)
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
    print(f"⚠ Checkpoint not exist: {args.checkpoint}", flush=True)
    print(f"Downloading from Google Drive...", flush=True)
    import gdown, torch, pathlib as P
    GDRIVE_FILE_ID = "1YNJ-HgasDS8bRDMAMKgYS0cs7H4-BP8R"
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, args.checkpoint, quiet=False)

assert os.path.exists(args.data_path), f'data path not exists: {args.data_path}'
assert os.path.exists(args.checkpoint), f'checkpoint not exists: {args.checkpoint}'
assert os.path.exists(args.output_sdf_path), f'output sdf path not exists: {args.output_sdf_path}'
predict_sdf_dir = os.path.join(args.output_sdf_path, args.task)
if not os.path.exists(predict_sdf_dir):
    os.mkdir(predict_sdf_dir)

complex_name_list = []
input_protein_list = []
input_ligand_list = []
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
    complex_name = data["complex_name"]
    input_protein = data["source_pdb"]
    assert os.path.exists(input_protein), f'input protein not exists: {input_protein}'
    input_ligand = sdf_or_mol(os.path.join(os.path.join(args.data_path, complex_name), f"{complex_name}_ligand"))
    if input_ligand is None and "source_sdf" in data.keys():
        input_ligand = data["source_sdf"]
    elif input_ligand is None:
        raise NotImplementedError
    complex_name_list.append(complex_name)
    input_protein_list.append(input_protein)
    input_ligand_list.append(input_ligand)
    print(f"Load sample {complex_name} with protein {input_protein} and ligand {input_ligand}.")
    
print("Test samples counts: ", len(complex_name_list))

lmdb_name = os.path.basename(args.test_lmdb).split(".")[0]
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
output_ligand_sdf = postprocessor.get_sdf(mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, complex_name_list, os.path.abspath(predict_sdf_dir), tta_times=args.conf_size)      

if args.use_flexible_docking != "rigid": # if apo
    pocket_dict = os.path.join(predict_sdf_dir, args.pocket_dict)
    input_protein =  get_general_predict_pdb(lmdb_file, pkl_file, args.batch_size, args.conf_size, predict_sdf_dir, args.max_pocket_atoms, pocket_dict)
    print("input_protein: ", input_protein)
else:
    input_protein = input_protein_list

# optimize
if not args.no_clash_fix:
    output_ligand_sdf, timeout_cases = postprocessor.clash_fix(output_ligand_sdf, input_protein, input_ligand_list)
    if timeout_cases:
        print(f"Warning: {len(timeout_cases)} ligands failed clash_fix (timeout or error), using original poses")
        print(f"Warning: Failed indices: {timeout_cases}")
    else:
        print("All ligands successfully processed by clash_fix")

print('output ligands path:\n', output_ligand_sdf)
end_time = time.time()
execution_time = end_time - start_time
print("Average time: ", execution_time/len(complex_name_list), "sec.")
print("Total time: ", execution_time, "sec.")
print('All processes done!')
