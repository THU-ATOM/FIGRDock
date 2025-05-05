import os
import argparse
import random
from rdkit import Chem
from predictor.processor import Processor, read_molecule, sdf_or_mol
import json
import numpy as np
import tqdm
import pandas as pd
# Command: python generate_lmdb.py --example --cluster

parser = argparse.ArgumentParser(description='generate lmdb for training')
parser.add_argument(
    "--data_path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/pdbbind_af2",
    help='data path for generating train and valid lmdb'
)
parser.add_argument(
    "--data_csv",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/pdbbind_af2/d3_with_clash_info.csv",
    help='data path for generating train and valid lmdb'
)
parser.add_argument(
    "--out_lmdb_dir", 
    type=str,
    default="/data/protein/BC_Data/Docking_Data/pdbbind_af2/bindnet_8A_all_with_Atom/",
    help='lmdb output directory'
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
    "--add_feature", 
    action='store_true',
)
parser.add_argument(
    "--esm_embeddings_dir", 
    help="where you store esm_embeddings_xxx.pt",
)
args = parser.parse_args()

preprocessor = Processor.build_processors(
    args.mode, args.nthreads, conf_size=args.conf_size, cluster=args.cluster,
    use_current_ligand_conf=args.use_current_ligand_conf)
preprocessor.error_directory_path=args.out_lmdb_dir

# Step1: get train/valid split
pdbbind_train_list = open("/data/protein/BC_Data/Docking_Data/pdbbind_af2/timesplit_train", "r").readlines()
pdbbind_valid_list = open("/data/protein/BC_Data/Docking_Data/pdbbind_af2/timesplit_val", "r").readlines()
pdbbind_test_list = open("/data/protein/BC_Data/Docking_Data/pdbbind_af2/timesplit_test", "r").readlines()
train_set = [t[:-1] for t in pdbbind_train_list]
valid_set = [v[:-1] for v in pdbbind_valid_list]
test_set = [v[:-1] for v in pdbbind_test_list]
if args.example:
    train_set = train_set[:200]
    valid_set = valid_set[:20]
    test_set = test_set[230:]
df = pd.read_csv(args.data_csv)

log_file_path = f"/data/protein/BC_Data/Docking_Data/pdbbind_af2/pdbbind_fail_debug.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
# Step2: preprocess and get output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid
def preprocess_pdbbind(data_set):
    output_ligand_name_list = []
    smiles_list = []
    input_protein_list = []
    input_ligand_list = []
    input_docking_grid_list = []

    for complex_name in tqdm.tqdm(data_set, "preprocessing data list"):
        data_info = df[df['pdb'] == complex_name]
        input_protein = [os.path.join(args.data_path, data_info['pdbFile'].item()[1:]),
                         os.path.join(args.data_path, data_info['af2File'].item()[1:]),]
        if not os.path.exists(input_protein[0]) or not os.path.exists(input_protein[1]): 
            open(log_file_path, "a").write(f"pdb not exist: {complex_name}\n")
            continue
        input_ligand = os.path.join(args.data_path, data_info['ligandFile'].item()[1:])
        mol = read_molecule(input_ligand)
        if mol == None: 
            open(log_file_path, "a").write(f"ligand not exists: {complex_name}\n")
            continue
        output_ligand_name_list.append(complex_name)
        smiles_list.append(None)
        input_protein_list.append(input_protein)
        input_ligand_list.append(input_ligand)
        input_docking_grid_list.append(None)
        
    return output_ligand_name_list, smiles_list, input_protein_list, input_ligand_list, input_docking_grid_list


for data_set, lmdb_name in [(test_set, "test_apo")]: # (train_set, "train"), (valid_set, "valid"), (test_set, "test_apo"), (test_set, "test_holo")
    preprocessor.lmdb_name = lmdb_name
    split = lmdb_name if lmdb_name in ["train", "valid", "test"] else "test"
    output_ligand_name_list, smiles_list, input_protein_list, \
        input_ligand_list, input_docking_grid_list = preprocess_pdbbind(data_set)
    print("Check: preprocess len", len(output_ligand_name_list))
    holo_protein_file_list = [p[0] for p in input_protein_list]
    start = 0
    end = 1000
    count = 0
    for i in range(1+(len(output_ligand_name_list)//1000)):
        if args.add_feature:
            preprocessor.read_esm_embeddings(split, args.esm_embeddings_dir, output_ligand_name_list, holo_protein_file_list, start, end)
        # Step3: write lmdb
        _, count = preprocessor.write_lmdb(output_ligand_name_list[start:end], smiles_list[start:end], input_protein_list[start:end], 
                                    input_ligand_list[start:end], input_docking_grid_list[start:end], seed=args.seed, 
                                    result_dir=args.out_lmdb_dir, use_sidechain="rigid", use_pdbbind=lmdb_name, pdbbind_add_feature=args.add_feature, start=count)
        start += 1000
        end += 1000
       