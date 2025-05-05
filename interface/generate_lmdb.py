import os
import argparse
import random
from rdkit import Chem
from predictor.processor import Processor

# Command: python generate_lmdb.py --example --cluster

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
    default="/data/protein/BC_Data/Uni-Mol/unimol_docking_v2/protein_ligand_binding_pose_prediction_v2_example/",
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
    "--use-sidechain",
    type=str,
    default="rigid",
    choices=["rigid", "consider_sidechain", "perturb_sidechain"],
    help='define pocket scope'
)

args = parser.parse_args()

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
print("Check: train:",len(train_set),"valid:", len(valid_set))
print("DEBUG: ", train_set[:5], valid_set[:5])

# Step2: preprocess and get output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid
def preprocess(complex_dir_list):
    output_ligand_name_list = []
    smiles_list = []
    input_protein_list = []
    input_ligand_list = []
    input_docking_grid_list = []
    
    ligand_name_list = []
    for complex_dir in complex_dir_list:
        complex_name = complex_dir.split("/")[-1]
        intput_protein = os.path.join(complex_dir, "protein.pdb")
        for _, _, files in os.walk(complex_dir):
            for name in files:
                if name.split(".")[1] == "sdf":
                    ligand_name = name.split(".")[0]
                    ligand_name_list.append(ligand_name)
        for ligand_name in ligand_name_list:
            output_ligand_name_list.append(complex_name+"-"+ligand_name)
            input_ligand = os.path.join(complex_dir, ligand_name+".sdf")
            input_ligand_list.append(input_ligand)
            supp = Chem.SDMolSupplier(input_ligand)
            mol = [mol for mol in supp if mol][0]
            ori_smiles = Chem.MolToSmiles(mol)
            smiles_list.append(ori_smiles)
            input_protein_list.append(intput_protein)
            input_docking_grid_list.append(os.path.join(complex_dir, ligand_name+".json"))
        ligand_name_list.clear()
    return output_ligand_name_list, smiles_list, input_protein_list, input_ligand_list, input_docking_grid_list


# Step3: write lmdb
preprocessor = Processor.build_processors(
    args.mode, args.nthreads, conf_size=args.conf_size, cluster=args.cluster,
    use_current_ligand_conf=args.use_current_ligand_conf)
for data_set, lmdb_name in [(train_set, "train"), (valid_set, "valid")]:
    preprocessor.lmdb_name = lmdb_name
    complex_dir_list = [os.path.join(args.data_path, _) for _ in data_set]
    output_ligand_name_list, smiles_list, input_protein_list, \
        input_ligand_list, input_docking_grid_list = preprocess(complex_dir_list)
    print("Check: preprocess len", len(output_ligand_name_list))
    _ = preprocessor.write_lmdb(output_ligand_name_list, smiles_list, input_protein_list, 
                                input_ligand_list, input_docking_grid_list, seed=args.seed, 
                                result_dir=args.out_lmdb_dir, use_sidechain=args.use_sidechain)

# moad set size: 34159
# complex size: 76695