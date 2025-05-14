import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task-name",
    type=str,
    default='gt_orig_ckpt_',
)
parser.add_argument(
    "--out-dir",
    type=str,
    default='/data/protein/BC_Data/Docking_Data/infer_posebusters/csv_pbvalid',
)
parser.add_argument(
    "--meta-csv",
    type=str,
    default='/data/protein/BC_Data/Docking_Data/unimol_posebusters_eval_sets/posebusters/posebuster_v2_set_meta.csv',
    help='provide basic information for ligand name and protein name'
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/unimol_posebusters_eval_sets/esm_posebusters/",
    help='data path of proteins and ligands'
)
parser.add_argument(
    "--apo-ligand-path",
    type=str,
    default="/data/protein/BC_Data/Docking_Data/infer/predict_sdf_posebuster428_grid10_ori_ckpt",
)
args = parser.parse_args()

import pandas as pd
import os
import subprocess
import tqdm

rows = [[
    "Name",
    "MOL_PRED loaded", 
    "MOL_TRUE loaded", 
    "MOL_COND loaded", 
    "Sanitization",
    "InChI convertible",
    "All atoms connected",
    "Molecular formula",
    "Molecular bonds",
    "Double bond stereochemistry",
    "Tetrahedral chirality",
    "Bond lengths",
    "Bond angles",
    "Internal steric clash",
    "Aromatic ring flatness",
    "Double bond flatness",
    "Internal energy",
    "Protein-ligand maximum distance",
    "Minimum distance to protein",
    "Minimum distance to organic cofactors",
    "Minimum distance to inorganic cofactors",
    "Minimum distance to waters",
    "Volume overlap with protein",
    "Volume overlap with organic cofactors",
    "Volume overlap with inorganic cofactors",
    "Volume overlap with waters",
    "RMSD ≤ 2Å",
    ]]

df = pd.read_csv(args.meta_csv)
fail_sanitize = 0
fail_others = 0
fail_rmsd = 0
pbvalid = 0

for i in tqdm.tqdm(range(len(df))):
    ligand_protein = df.loc[i]
    lig_code = ligand_protein["lig_code"]
    pdb_code = ligand_protein["pdb_code"]
    holo_ligand_path = os.path.join(args.data_path, f"{pdb_code}_{lig_code}", f"{pdb_code}_{lig_code}_ligand.sdf")
    protein_path = os.path.join(args.apo_ligand_path, f"{pdb_code}_predict.pdb")
    apo_ligand_path = os.path.join(args.apo_ligand_path, f"{pdb_code}_{lig_code}.sdf")
    if not os.path.exists(holo_ligand_path): 
        print("holo not exist for: ", holo_ligand_path)
        continue
    assert os.path.exists(holo_ligand_path)
    assert os.path.exists(protein_path)
    
    
    out_fmt = "short"
    posebusters_cli_command = f"bust {apo_ligand_path} -l {holo_ligand_path} -p {protein_path} --outfmt {out_fmt}"
    
    result = subprocess.run([posebusters_cli_command], stdout=subprocess.PIPE, text=True, shell=True)
    
    if len(result.stdout) == 0:
        fail_sanitize += 1
    else:
        try:
            if int(result.stdout[-9:-7]) == 26:
                pbvalid += 1
            else:
                out_fmt = "long"
                posebusters_cli_command = f"bust {apo_ligand_path} -l {holo_ligand_path} -p {protein_path} --outfmt {out_fmt}"
                result = subprocess.run([posebusters_cli_command], stdout=subprocess.PIPE, text=True, shell=True)
                print(i, "fail_pb_valid") # DEBUG: print result.stdout
                out_lines = result.stdout.split("\n")
                row = [f"{pdb_code}_{lig_code}"]
                for out_line in out_lines[1:-1]:
                    if out_line[-4:] == "Fail":
                        row.append(1)
                        if out_line[:4] == "RMSD":
                            fail_rmsd += 1
                    else:
                        row.append(0)
                fail_others += 1
                rows.append(row)
        except:
            print("ERROR:", i, result.stdout)


print("success", pbvalid)
print("fail_sanitize", fail_sanitize)
print("fail_rmsd", fail_rmsd)
print("fail_all", fail_others)
    # rows.append([
    #     protein,
    #     protein_path,
    #     holo_ligand_path,
    #     apo_ligand_path
    # ])
    

import csv

out_csv = os.path.join(args.out_dir, f"{args.task_name}.csv")
# 打开文件并写入
with open(out_csv, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
    
import pandas as pd

df = pd.read_csv(out_csv)
column_sums = df.sum()
out_log = os.path.join(args.out_dir, f"{args.task_name}.log")
column_sums.to_string(out_log, index=True)
with open(out_log, "a") as file:
    file.write(f"success: {pbvalid}\n")
    file.write(f"fail_sanitize: {fail_sanitize}\n")
    file.write(f"fail_rmsd: {fail_rmsd}\n")
    file.write(f"fail_pbvalid: {fail_others-fail_rmsd}\n")