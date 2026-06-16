# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import lmdb
import pickle
import copy
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMolConformers
from biopandas.pdb import PandasPdb
import warnings
from Bio.PDB import PDBParser
biopython_parser = PDBParser()
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Selection import unfold_entities
from flexible_docking_utils import process_sc_rawdata, check_res_intact, SORTING_DICT
from pdbbind_benchmark_sc_utils import allowable_features, bond_type_to_value
from collections import defaultdict
import torch
import subprocess
from multiprocessing import Pool
import traceback

def parse_pdb_from_path(path):
    ret = parse_pdb_structure_from_path(path)
    return ret[0]


def parse_pdb_structure_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure(os.path.basename(path), path)
        return structure

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1

import contextlib
@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    seed_origin = seed
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))
    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol

def sdf_or_mol(path):
    for suffix in ['.sdf', '.mol2']:
        cur_path = f"{path}{suffix}"
        if not os.path.exists(cur_path):
            continue
        if read_molecule(cur_path, sanitize=True, remove_hs=False) != None:
            return cur_path
    return None


class Processor:
    def __init__(self, 
        mode:str='single', 
        nthreads:int=8, 
        conf_size:int=10, 
        cluster:bool=False, 
        main_atoms:List[str]=["N", "CA", "C", "O", "H"], 
        allow_pocket_atoms:List[str]=[['C', 'H', 'N', 'O', 'S']],
        use_current_ligand_conf:bool=False,
        error_directory_path: str=""
    ):
        self.mode = mode
        self.nthreads = nthreads
        self.conf_size = conf_size
        self.cluster = cluster
        self.main_atoms = main_atoms
        self.allow_pocket_atoms = allow_pocket_atoms
        if self.mode in ['batch_one2one', 'batch_one2many']:
            self.lmdb_name = 'batch_data'
        self.use_current_ligand_conf = use_current_ligand_conf
        self.error_directory_path = error_directory_path

    def preprocess(self, input_protein:str, input_ligand, input_docking_grid:str, output_ligand_name:str, out_lmdb_dir:str, use_sidechain:str):
        seed = 42 
        if self.mode=='single':
            supp = Chem.SDMolSupplier(input_ligand)
            mol = [mol for mol in supp if mol][0]
            ori_smiles = Chem.MolToSmiles(mol)
            smiles_list = [ori_smiles]
            input_protein = [input_protein]
            input_ligand = [input_ligand]
            input_docking_grid = [input_docking_grid]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            if self.mode == 'batch_one2many':
                input_protein = [input_protein] * len(input_ligand)
            smiles_list = []
            for i in range(len(input_ligand)):
                supp = Chem.SDMolSupplier(input_ligand[i])
                mol = [mol for mol in supp if mol][0]
                ori_smiles = Chem.MolToSmiles(mol)
                smiles_list.append(ori_smiles)
        lmdb_name = self.write_lmdb(output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=seed, result_dir=out_lmdb_dir, use_sidechain=use_sidechain)
        return lmdb_name

    def single_conf_gen(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=8 # maxAttempts=5000,
        )
        sz = len(allconformers)
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                continue
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def single_conf_gen_no_MMFF(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=8 # maxAttempts=5000
        )
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def clustering_coords(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        rdkit_coords_list = []
        if not cluster:
            M = N
        if gen_mode == 'mmff':
            rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
        elif gen_mode == 'no_mmff':
            rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
        noHsIds = [
            rdkit_mol.GetAtoms()[i].GetIdx()
            for i in range(len(rdkit_mol.GetAtoms()))
            if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
        ]
        ### exclude hydrogens for aligning
        AlignMolConformers(rdkit_mol, atomIds=noHsIds)
        sz = len(rdkit_mol.GetConformers())
        for i in range(sz):
            _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            rdkit_coords_list.append(_coords)

        ### exclude hydrogens for clustering, pick closest to centroid:
        if cluster:
            # (num_confs, num_atoms, 3)
            rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
            # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
            rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
            kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten)
            # (num_clusters, num_atoms, 3)
            center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
            # (num_cluster, num_confs)
            cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
            # (num_confs,)
            argmin = np.argmin(cdist, axis=-1)
            coords_list = [rdkit_coords_list[i] for i in argmin]
        else:
            coords_list = rdkit_coords_list

        return coords_list

    def find_residues_in_pocket(self, pocket: dict, pdf):
        """
        Given a pocket config and a residue df, 
        return a list of residues that are in the pocket
        """
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
        min_x, max_x = _get_vertex(pocket, "x")
        min_y, max_y = _get_vertex(pocket, "y")
        min_z, max_z = _get_vertex(pocket, "z")
        min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
        max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
        patoms, pcoords, residues, restypes = [], np.empty((0,3)), [], []
        for i in range(len(pdf)):
            atom_info = pdf.iloc[i]
            _rescoor = np.array(atom_info[['x_coord','y_coord','z_coord']].values).reshape(-1,3)
            mapping = (_rescoor > min_array) & (_rescoor < max_array)
            if (mapping.sum(-1) == 3).sum() > 0:
                patoms += [atom_info['atom_name']]
                pcoords = np.concatenate((pcoords, _rescoor), axis=0)
                residues += [str(atom_info['chain_id'])+"_"+str(atom_info['residue_number'])]
                restypes += [str(atom_info['residue_name'])]
        return patoms, pcoords, residues, restypes

    def extract_pocket(self, input_protein, input_docking_grid):
        try:
            pmol = PandasPdb().read_pdb(input_protein)
        except:
            with open('failed_pocket.txt', 'a') as f:
                f.write(' '.join(input_protein)+'\n')
            return None
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)

        pdf = pmol.df['ATOM']
        patoms, pcoords, residues, restypes = self.find_residues_in_pocket(box_dict, pdf)
        def _filter_pocketatoms(atom):
            if atom[:2] in ['Cd','Cs', 'Cn', 'Ce', 'Cm', 'Cf', 'Cl', 'Ca', 'Cr', 'Co', 'Cu', 'Nh', 'Nd', 'Np', 'No', 'Ne', 'Na', 'Ni', \
                'Nb', 'Os', 'Og', 'Hf', 'Hg', 'Hs', 'Ho', 'He', 'Sr', 'Sn', 'Sb', 'Sg', 'Sm', 'Si', 'Sc', 'Se']:
                return None
            if atom[0] >= '0' and atom[0] <= '9':
                return _filter_pocketatoms(atom[1:])
            if atom[0] in ['Z','M','P','D','F','K','I','B']:
                return None
            if atom[0] in self.allow_pocket_atoms:
                return atom
            return atom

        atoms, index, residues_tmp = [], [], []
        for i,a in enumerate(patoms):
            output = _filter_pocketatoms(a)
            if output is not None:
                index.append(True)
                atoms.append(output)
                residues_tmp.append(residues[i])
            else:
                index.append(False)
        coordinates = pcoords[index].astype(np.float32)
        residues = residues_tmp
        patoms = atoms
        pcoords = [coordinates]
        side = [0 if a in self.main_atoms else 1 for a in patoms]
        return patoms, pcoords, residues, restypes, side, box_dict

    def parser(self, content):
        smiles, input_protein, input_ligand, input_docking_grid, seed = content
        patoms, pcoords, residues, restypes, side, config = self.extract_pocket(input_protein, input_docking_grid)
        # get ground truth conformation and generate ligand conformation
        # supp = Chem.SDMolSupplier(input_ligand)
        # mol = [mol for mol in supp if mol][0]
        file_type = input_ligand.split(".")[-1]
        if file_type == 'mol2':
            mol = Chem.MolFromMol2File(input_ligand)
        elif file_type == 'sdf':
            supplier = Chem.SDMolSupplier(input_ligand)
            mol = supplier[0]
        if self.use_current_ligand_conf:
            return pickle.dumps(
                {
                    "atoms": [atom.GetSymbol() for atom in mol.GetAtoms()],
                    "coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                    "mol_list": [mol],
                    "pocket_atoms": patoms,
                    "pocket_coordinates": pcoords,
                    "side": side,
                    "residue": residues,
                    "config": config,
                    "holo_coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                    "holo_mol": mol,
                    "holo_pocket_coordinates": pcoords,
                    "smi": smiles,
                    "pocket": input_protein,
                },
                protocol=-1,
            )
        mol = Chem.AddHs(mol)
        smiles = Chem.MolToSmiles(mol)
        latoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        holo_coordinates = [mol.GetConformer().GetPositions().astype(np.float32)]
        holo_mol = mol
        N = self.conf_size
        M = self.conf_size * 10
        mol_list = [mol] * N
        try:
            coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='mmff') 
        except:
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='no_mmff') 
            except:
                print(f'Failed to generate conformers with RDKit: {input_ligand}, skipped!')  
                return None
            # except:
            #     def fix_fail_init(mol): 
            #         # rdkit coords could not be generated without using random coords. using random coords now.
            #         ps = AllChem.ETKDGv2()
            #         ps.useRandomCoords = True
            #         AllChem.EmbedMolecule(mol, ps)
            #         AllChem.MMFFOptimizeMolecule(mol, confId=0)
            #         lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            #         return lig_coords
            #     try:
            #         error_pocket_file = open(os.path.join(self.error_directory_path,"ligand_init_fail.log"), "a")
            #         error_pocket_file.write(f"{input_ligand}: use fix fail init\n")
            #         coordinate_list = []
            #         for i in range(10):
            #             coordinate_list.append(fix_fail_init(mol))
            #     except:
            #         print(f'Failed to generate conformers with RDKit: {input_ligand}, skipped!')  
            #         error_pocket_file = open(os.path.join(self.error_directory_path,"ligand_init_fail.log"), "a")
            #         error_pocket_file.write(f"{input_ligand} fail\n")
            #         return None
        with open("./test_script/log/debug_parser_pocket_size.log","a") as file:
            res_ = list(set([res for res in residues]))
            file.write(input_protein+": "+str(len(patoms))+" "+str(len(res_))+"\n")
        return pickle.dumps(
            {
                "atoms": latoms,
                "coordinates": coordinate_list,
                "mol_list": mol_list,
                "pocket_atoms": patoms,
                "pocket_coordinates": pcoords,
                "side": side,
                "restype": restypes,
                "residue": residues,
                "config": config,
                "holo_coordinates": holo_coordinates,
                "holo_mol": holo_mol,
                "holo_pocket_coordinates": pcoords,
                "smi": smiles,
                "pocket": input_protein,
            },
            protocol=-1,
        )

    def write_lmdb(self, output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=42, result_dir="./results", use_sidechain=None, use_pdbbind=False, pdbbind_add_feature=False, start=0):
        os.makedirs(result_dir, exist_ok=True)
        if self.mode == 'single':
            outputfilename = os.path.join(result_dir, output_ligand_name + ".lmdb")
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            outputfilename = os.path.join(result_dir, self.lmdb_name + ".lmdb")
            output_ligand_name = self.lmdb_name
        # try:
        #     os.remove(outputfilename)
        # except:
        #     pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(50e9),
        )
        txn_write = env_new.begin(write=True)
        print("Start preprocessing data...")
        print(f'Number of ligands: {len(smiles_list)}')
        np.random.seed(seed)
        seed = [seed] * len(input_ligand)
        if use_sidechain!="rigid" and not use_pdbbind:
            use_sidechain_list = [use_sidechain] * len(input_ligand)
            content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed, use_sidechain_list)
            parse_func = self.parser_sidechain
        elif use_pdbbind:
            istest = [use_pdbbind] * len(input_ligand)
            pdbbind_add_feature = [pdbbind_add_feature] * len(input_ligand)
            content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed, istest, pdbbind_add_feature)
            parse_func = self.parser_pdbbind 
        else:
            content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed)
            parse_func = self.parser
        with Pool(self.nthreads) as pool:
            i = start
            failed_num = 0
            for inner_output in tqdm(pool.imap(parse_func, content_list)):
                if inner_output is not None:
                    if i % 100 == 0:
                        if i != 0:  # 避免在i=0时提交空事务
                            txn_write.commit()
                            with open(os.path.join(self.error_directory_path, "process_lmdb.log"), "a") as file:
                                file.write(f"finish:{i} / fail_num: {failed_num} \n")
                            txn_write = env_new.begin(write=True)
                    txn_write.put(f"{i}".encode("ascii"), inner_output)
                    i+=1
                elif inner_output is None: 
                    failed_num += 1
            if i % 100 != 0:  # 避免在i是100的倍数时重复提交
                txn_write.commit()
            env_new.close()
        print(f'Total num: {len(smiles_list)}, Success: {i}, Failed: {failed_num}')
        print("Done!")
        return output_ligand_name, i

    def load_lmdb_data(self, lmdb_path, key):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        _keys = list(txn.cursor().iternext(values=False))
        collects = []
        for idx in range(len(_keys)):
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            collects.append(data[key])
        return collects

    def postprocess_data_pre(self, predict_file, lmdb_file):
        mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        predict = pd.read_pickle(predict_file)
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]
        for batch in predict:
            sz = batch['atoms'].size(0)
            for i in range(sz):
                smi_list.append(batch['smi_name'][i])
                pocket_list.append(batch['pocket_name'][i])
                prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))
                
                token_mask = batch['atoms'][i]>2

                holo_coordinates = batch['holo_coordinates'][i]
                holo_coordinates = holo_coordinates[token_mask,:]
                holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                coord_predict = batch['coord_predict'][i]
                coord_predict = coord_predict[token_mask,:]
                coord_predict = coord_predict.numpy().astype(np.float32)

                holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                holo_center_coordinates.numpy().astype(np.float32)

                holo_center_coords_list.append(holo_center_coordinates)        
                coords_predict_list.append(coord_predict)
                holo_coords_list.append(holo_coordinates)

        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list

    def set_coord(self, mol, coords):
        for i in range(coords.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
        return mol

    def add_coord(self, mol, xyz):
        x, y, z = xyz
        conf = mol.GetConformer(0)
        pos = conf.GetPositions()
        pos[:, 0] += x
        pos[:, 1] += y
        pos[:, 2] += z
        for i in range(pos.shape[0]):
            conf.SetAtomPosition(
                i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
            )
        return mol
    
    def get_sdf(self, mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, tta_times=10):
        print("Start converting model predictions into sdf files...")
        output_ligand_list = []
        if self.mode == 'single':
            output_ligand_name = [output_ligand_name]
        for i in tqdm(range(len(smi_list)//tta_times)):
            coords_predict_tta = coords_predict_list[i*tta_times:(i+1)*tta_times]
            prmsd_score_tta = prmsd_score_list[i*tta_times:(i+1)*tta_times]
            mol_list_tta = mol_list[i*tta_times:(i+1)*tta_times]
            holo_center_coords_tta = holo_center_coords_list[i*tta_times:(i+1)*tta_times]
            idx = np.argmin(prmsd_score_tta)
            bst_predict_coords = coords_predict_tta[idx]
            mol = mol_list_tta[idx]
            mol = self.set_coord(mol, bst_predict_coords)
            holo_center_coords = holo_center_coords_tta[idx]
            mol = self.add_coord(mol, holo_center_coords.numpy())
            os.makedirs(output_ligand_dir, exist_ok=True)
            outputfilename = os.path.join(output_ligand_dir, str(output_ligand_name[i]) + '.sdf')
            try:
                os.remove(outputfilename)
            except:
                pass
            Chem.MolToMolFile(mol, outputfilename)
            output_ligand_list.append(outputfilename)
        print("Done!")
        if self.mode == 'single':
            return output_ligand_list[0]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            return output_ligand_list
    
    def single_clash_fix(self, input_content):
        input_ligand, output_ligand, label_ligand, pocket_mol = input_content
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "figrdock", "scripts", "6tsr.py")
        cmd = [
            '/opt/conda/bin/python',
            script_path,
            '--input-ligand',  input_ligand,
            '--output-ligand', output_ligand,
            '--label-ligand',  label_ligand,
            '--pocket-mol',    pocket_mol,
            '--num-6t-trials', '5'
        ]
        subprocess.run(cmd, check=True, timeout=1200)
        return True
    
    def _safe_single_clash_fix(self, args):
        """子进程里跑真正的 single_clash_fix，并捕获所有异常"""
        idx, item = args
        try:
            self.single_clash_fix(item)      # 原来函数返回 True/False 都行
            return idx, True, None
        except subprocess.TimeoutExpired:
            return idx, False, 'TIMEOUT'
        except Exception as e:
            # 把异常信息带回来，方便调试
            return idx, False, f'{type(e).__name__}:{e}|{traceback.format_exc()}'

    def clash_fix(self, predicted_ligand, input_protein, input_ligand):
        if self.mode == 'batch_one2many':
            input_protein = [input_protein] * len(input_ligand)
        elif self.mode == 'single':
            input_ligand = [input_ligand]
            input_protein = [input_protein]
            predicted_ligand = [predicted_ligand]
        input_content = list(zip(predicted_ligand, predicted_ligand, input_ligand, input_protein))

        timeout_cases = []
        with Pool(self.nthreads) as pool:
            # 把“带索引”的输入丢给安全壳函数
            results = pool.imap(
                self._safe_single_clash_fix,
                enumerate(input_content),
                chunksize=1
            )
            for idx, success, flag in tqdm(results, total=len(input_content)):
                if success:
                    continue                 # 正常，什么都不用干
                if flag == 'TIMEOUT':
                    print(f'[TIMEOUT] 第 {idx} 条任务超时，输入为：{input_content[idx]}')
                else:
                    print(f'[ERROR] 第 {idx} 条任务失败：{flag}')
                timeout_cases.append(idx)

        return predicted_ligand, timeout_cases

    @classmethod
    def build_processors(
        cls, 
        mode='single', 
        nthreads = 8, 
        conf_size = 10, 
        cluster=False,
        use_current_ligand_conf:bool=False
    ):
        return cls(
            mode, 
            nthreads, 
            conf_size=conf_size, 
            cluster=cluster, 
            use_current_ligand_conf=use_current_ligand_conf
        )
        
    def process_ligand(self, input_ligand, seed, file_type="sdf", use_lig_feature=False):
        # read mol from sdf file
        if file_type == 'mol2':
            ligand = Chem.MolFromMol2File(input_ligand)
        elif file_type == 'sdf':
            supplier = Chem.SDMolSupplier(input_ligand)
            ligand = supplier[0]
        else:
            assert NotImplementedError
        smiles = Chem.MolToSmiles(ligand)
        ligand = Chem.AddHs(ligand)
        # get position of ligand
        ligand_pos = ligand.GetConformer().GetPositions().astype(np.float32)
        # get ligand atom type
        ligand_atoms_list = [atom.GetSymbol() for atom in ligand.GetAtoms()]
        ligand_center = np.mean(ligand_pos, axis=0)
        ligand_sphere_radius = np.sqrt(np.sum(np.square(ligand_pos - ligand_center), axis=1)).mean() # mean to max?
        
        holo_ligand = ligand
        N = self.conf_size
        M = self.conf_size * 10
        mol_list = [ligand] * N
        try:
            coordinate_list = self.clustering_coords(ligand, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='mmff') 
            assert len(coordinate_list) != 0
            for i in range(10-len(coordinate_list)):
                coordinate_list.append(coordinate_list[i%len(coordinate_list)])
        except:
            try:
                coordinate_list = self.clustering_coords(ligand, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='no_mmff') 
                assert len(coordinate_list) != 0
                for i in range(10-len(coordinate_list)):
                    coordinate_list.append(coordinate_list[i%len(coordinate_list)])
            # except:
            #     print(f'Failed to generate conformers with RDKit: {input_ligand}, skipped!')  
            #     error_pocket_file = open(os.path.join(self.error_directory_path,"ligand_init_fail.log"), "a")
            #     error_pocket_file.write(f"{input_ligand}\n")
            #     return None
            except:
                def fix_fail_init(mol): 
                    # rdkit coords could not be generated without using random coords. using random coords now.
                    ps = AllChem.ETKDGv2()
                    ps.useRandomCoords = True
                    AllChem.EmbedMolecule(mol, ps)
                    AllChem.MMFFOptimizeMolecule(mol, confId=0)
                    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
                    return lig_coords
                try:
                    error_pocket_file = open(os.path.join(self.error_directory_path,"ligand_init_fail.log"), "a")
                    error_pocket_file.write(f"{input_ligand}: use fix fail init\n")
                    coordinate_list = []
                    for i in range(10):
                        coordinate_list.append(fix_fail_init(ligand))
                except:
                    print(f'Failed to generate conformers with RDKit: {input_ligand}, skipped!')  
                    error_pocket_file = open(os.path.join(self.error_directory_path,"ligand_init_fail.log"), "a")
                    error_pocket_file.write(f"{input_ligand}\n")
                    return None
        
        if use_lig_feature:
            ringinfo = ligand.GetRingInfo()
            atom_features_list = []
            L = ligand.GetNumAtoms()
            bond_type_matrix = [[-1 for _ in range(L)] for _ in range(L)]
            for idx1, atom1 in enumerate(ligand.GetAtoms()):
                #g_charge = atom.GetDoubleProp('_GasteigerCharge')
                atom_features_list.append([
                    safe_index(allowable_features['possible_atomic_num_list'], atom1.GetAtomicNum()),
                    allowable_features['possible_chirality_list'].index(str(atom1.GetChiralTag())),
                    safe_index(allowable_features['possible_degree_list'], atom1.GetTotalDegree()),
                    safe_index(allowable_features['possible_formal_charge_list'], atom1.GetFormalCharge()),
                    safe_index(allowable_features['possible_implicit_valence_list'], atom1.GetImplicitValence()),
                    safe_index(allowable_features['possible_numH_list'], atom1.GetTotalNumHs()),
                    safe_index(allowable_features['possible_number_radical_e_list'], atom1.GetNumRadicalElectrons()),
                    safe_index(allowable_features['possible_hybridization_list'], str(atom1.GetHybridization())),
                    allowable_features['possible_is_aromatic_list'].index(atom1.GetIsAromatic()),
                    safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx1)),
                    allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 3)),
                    allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 4)),
                    allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 5)),
                    allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 6)),
                    allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 7)),
                    allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx1, 8)),
                    #g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
                ])
                for idx2, atom2 in enumerate(ligand.GetAtoms()):
                    bond = ligand.GetBondBetweenAtoms(idx1, idx2)
                    if bond:
                        bond_type = bond.GetBondType()
                        bond_type_matrix[idx1][idx2] = bond_type_to_value.get(bond_type, -1)  # 如果没有匹配的键，返回 -1
            return ligand_atoms_list, coordinate_list, mol_list, ligand_pos, holo_ligand, smiles, torch.tensor(atom_features_list), torch.tensor(bond_type_matrix)
        return ligand_atoms_list, coordinate_list, mol_list, ligand_pos, holo_ligand, smiles, None, None
    
    def process_pocket(self, rec, holo_rec, residue_ids, input_protein, complex_name=None, align_mistake=""):
        res_info_dict = {resnum: defaultdict(list) for resnum in residue_ids}
        if align_mistake == "exp_larger":
            apo_holo_atom_pair_list = zip(rec.get_atoms(), list(holo_rec.get_atoms())[-1])
        elif align_mistake == "comp_larger":
            apo_holo_atom_pair_list = zip(list(rec.get_atoms())[-1], holo_rec.get_atoms())
        else:
            apo_holo_atom_pair_list = zip(rec.get_atoms(), holo_rec.get_atoms())
        if complex_name is not None:
            esm_emb = self.esm_embeddings[complex_name]
            esm_emb = np.concatenate(esm_emb, axis=0)
            if not len(esm_emb) == len(unfold_entities(holo_rec, "R")):
                print(complex_name, f"{len(esm_emb)}/{len(unfold_entities(rec, 'R'))}")
            res_list = [f'{res.get_parent().get_id()}_{res.get_id()[1]}{res.get_id()[2]}'.strip() for res in rec.get_residues()]

        for all_atom, holo_all_atom in apo_holo_atom_pair_list:
            residu_id = f'{all_atom.get_parent().get_parent().get_id()}_{all_atom.get_parent().get_id()[1]}{all_atom.get_parent().get_id()[2]}'.strip()
            if residu_id in res_info_dict:
                pocket_atom = all_atom.get_name()
                pocket_restype = all_atom.get_parent().get_resname()
                pocket_restype_token = safe_index(allowable_features['possible_amino_acids'], all_atom.get_parent().get_resname())
                coordinates = list(all_atom.get_coord())
                holo_coordinates = list(holo_all_atom.get_coord())
                res_info_dict[residu_id]["restypes"].append(pocket_restype)
                res_info_dict[residu_id]["restype_tokens"].append(pocket_restype_token)
                res_info_dict[residu_id]["atoms"].append(pocket_atom)
                res_info_dict[residu_id]["coords"].append(coordinates)
                res_info_dict[residu_id]["resnums"].append(residu_id)
                res_info_dict[residu_id]["holo_coords"].append(holo_coordinates)
                if complex_name is not None:
                    res_rank = res_list.index(residu_id)
                    if res_rank < len(esm_emb):
                        res_info_dict[residu_id]["lm_embeddings"].append(esm_emb[res_rank])
                    else:
                        res_info_dict[residu_id]["lm_embeddings"].append(np.zeros_like(esm_emb[-1], dtype=np.float32))
        
        # check side chain intact
        pocket_info_dict = {
            "resnums": [],
            "restypes": [],
            "atoms": [],
            "coords": [],
            "holo_coords": [],
            "restype_tokens": [],
        }
        if complex_name is not None:
            pocket_info_dict["lm_embeddings"] = []

        for resnum in residue_ids:
            filtered_res_dict = check_res_intact(res_info_dict[resnum], complex_name)
            if filtered_res_dict is None:
                print(f'check pocket {input_protein}, residue {resnum} failed')
                # return None
            else:
                pocket_info_dict["resnums"].extend(filtered_res_dict["resnums"])
                pocket_info_dict["restypes"].extend(filtered_res_dict["restypes"])
                pocket_info_dict["restype_tokens"].extend(filtered_res_dict["restype_tokens"])
                pocket_info_dict["atoms"].extend(filtered_res_dict["atoms"])
                pocket_info_dict["coords"].extend(filtered_res_dict["coords"])
                pocket_info_dict["holo_coords"].extend(filtered_res_dict["holo_coords"])
                if complex_name is not None:
                    pocket_info_dict["lm_embeddings"].extend(filtered_res_dict["lm_embeddings"])

        pocket_info_dict["coords"] = [np.array(pocket_info_dict["coords"])]
        pocket_info_dict["holo_coords"] = [np.array(pocket_info_dict["holo_coords"])]
        
        return pocket_info_dict
        
    def parser_sidechain(self, content):
        smiles, input_protein, input_ligand, input_docking_grid, seed, use_sidechain = content
        
        # filter out invalid pocket input
        try:
            rec = parse_pdb_from_path(input_protein)
        except:
            error_pocket_file = open(os.path.join(self.error_directory_path, "parse_sidechain_pocket_fail.log"), "a")
            error_pocket_file.write(input_protein)
            return None
        
        ligand_ret = self.process_ligand(input_ligand, seed)
        if ligand_ret == None:
            return None
        else:
            ligand_atoms_list, coordinate_list, mol_list, ligand_pos, holo_ligand, smiles = ligand_ret
        
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)
        
        def _get_vertex(pocket: dict, axis: str) -> tuple:
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
        
        # Process pocket
        pocket_info_dict = self.process_pocket(rec, rec, residue_ids, input_protein)
        
        # Less atoms in this setting is due to not considering Hydrogen which is removed in training.
        # with open(log_file, "a") as file:
        #     file.write(str(len(pocket_info_dict["atoms"]))+ " " +pdb_file+"\n")
        if use_sidechain == "perturb_sidechain":
            with numpy_seed(seed, input_protein):
                pocket_coordinates = process_sc_rawdata(pocket_info_dict, new_mask=False)
            pocket_coordinates = [np.array(pocket_coordinates)]
        elif use_sidechain == "consider_sidechain":
            pocket_coordinates = pocket_info_dict["coords"]
        else:
            assert NotImplementedError
        
        new_pocket_info_dict = {
            "resnums": [],
            "restypes": [],
            "atoms": [],
            "coords": [],
            "holo_coords": [],
        }
        
        for idx, _rescoor in enumerate(pocket_coordinates[0]):
            mapping = (_rescoor > min_array) & (_rescoor < max_array)
            if (mapping.sum(-1) == 3).sum() > 0:
                new_pocket_info_dict["restypes"].append(pocket_info_dict["restypes"][idx])
                new_pocket_info_dict["atoms"].append(pocket_info_dict["atoms"][idx])
                new_pocket_info_dict["coords"].append(pocket_info_dict["coords"][0][idx])
                new_pocket_info_dict["resnums"].append(pocket_info_dict["resnums"][idx])
                new_pocket_info_dict["holo_coords"].append(_rescoor)
        new_pocket_info_dict["holo_coords"] = [np.array(new_pocket_info_dict["holo_coords"])]
        new_pocket_info_dict["coords"] = [np.array(new_pocket_info_dict["coords"])]
        
        with open(f"./debug_{use_sidechain.split('_')[0]}2.log","a") as file:
            # res_ = list(set([f"{t}_{n}" for t,n in zip(new_pocket_info_dict["restypes"], new_pocket_info_dict["resnums"])]))
            res_ = list(set([res for res in new_pocket_info_dict["resnums"]]))
            file.write(input_protein+": "+str(len(new_pocket_info_dict["atoms"]))+" "+str(len(res_))+"\n")
        
        return pickle.dumps({
            "atoms": ligand_atoms_list,
            "coordinates": coordinate_list,
            "mol_list": mol_list,
            "pocket_atoms": new_pocket_info_dict["atoms"], 
            "pocket_coordinates": new_pocket_info_dict["coords"],
            "restype": new_pocket_info_dict["restypes"],
            # "side": side,
            "residue": new_pocket_info_dict["resnums"],
            "config": box_dict,
            "holo_coordinates": [ligand_pos],
            "holo_mol": holo_ligand,
            "holo_pocket_coordinates": new_pocket_info_dict["holo_coords"],
            "smi": smiles,
            "pocket": input_protein,
        }, protocol=-1,)
        
    def parser_pdbbind(self, content, threshold=600):
        smiles, input_protein, input_ligand, input_docking_grid, seed, istest, add_feature = content
        complex_name = input_protein[0].split("/")[-1].split("_")[0]
        
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
        
        # filter out invalid pocket input
        # align apo and holo
        step = "start"
        align_mistake = ""
        try:
            experimental_receptor = parse_pdb_from_path(input_protein[0]) # holo protein
            computational_receptor = parse_pdb_from_path(input_protein[1]) # apo protein
            step = "parse success"
            _remove_hs(experimental_receptor)
            _sort_atoms_by_element(experimental_receptor)
            _remove_hs(computational_receptor)
            _sort_atoms_by_element(computational_receptor)
            step = "remove and sort success"
            if len(list(computational_receptor.get_atoms())) != len(list(experimental_receptor.get_atoms())): 
                if len(list(experimental_receptor.get_atoms())) - len(list(computational_receptor.get_atoms())) == 1:
                    if [a.name for a in computational_receptor.get_atoms()] != [a.name for a in experimental_receptor.get_atoms()][:-1]:
                        print(f'Failed align receptor: {complex_name} atom names, skipped!')  
                        return None
                    align_mistake = "exp_larger"
                elif len(list(experimental_receptor.get_atoms())) - len(list(computational_receptor.get_atoms())) == -1:
                    if [a.name for a in computational_receptor.get_atoms()][:-1] != [a.name for a in experimental_receptor.get_atoms()]:
                        print(f'Failed align receptor: {complex_name} atom names, skipped!')  
                        return None
                    align_mistake = "comp_larger"
                else:
                    print(f'Failed align receptor: apo-{len(list(computational_receptor.get_atoms()))}, holo-{len(list(experimental_receptor.get_atoms()))}, skipped!') 
                    return None
            if align_mistake == "" and [a.name for a in computational_receptor.get_atoms()] != [a.name for a in experimental_receptor.get_atoms()]:
                print(f'Failed align receptor: {complex_name} atom names, skipped!')  
                return None
            step = "align sucess"
        except:
            error_pocket_file = open(os.path.join(self.error_directory_path, "pdbbind_pocket_fail.log"), "a")
            print("DEBUG: error_pocket", f"{input_protein[0]}: {step}\n")
            error_pocket_file.write(f"{input_protein[0]}: {step}\n")
            return None
        # Process ligand:
        # Save holo conformation to holo_mol, ligand_pos
        # Generate apo conformation via rdkit and save to mol_list, coordinate_list 
        ligand_file_type =input_ligand.split(".")[-1]
        ligand_ret = self.process_ligand(input_ligand, seed, file_type=ligand_file_type, use_lig_feature=add_feature)
        if ligand_ret == None:
            return None
        else:
            ligand_atoms_list, coordinate_list, mol_list, ligand_pos, holo_ligand, smiles, atom_features_list, bond_type_matrix = ligand_ret
        
        # Process pocket
        # Step 1:
        # use holo to decide pocket center and pocket radius
        # decide apo pocket scope
        # choose apo/holo as input receptor according to align rmsd
        
        def _calculate_binding_pocket(receptor, ligand, buffer, pocket_cutoff=5):
            d = torch.cdist(receptor, ligand)
            label = torch.any(d < pocket_cutoff, axis=1)
            if label.any():
                center_pocket = receptor[label].mean(axis=0)
            else:
                center_pocket = receptor[d.min(axis=1)[0].argmin()]
            radius_pocket = torch.linalg.norm(ligand - center_pocket[None, :], axis=1)
            return center_pocket.cpu().numpy(), (radius_pocket.max() + buffer).cpu().numpy()
        
        def pocket_selector(residue, pocket_center, pocket_radius):
            return (np.linalg.norm(np.array([a.coord for a in residue.child_list]) - pocket_center, axis=1) < pocket_radius).any()
            
        def RMSD(atom_ids, atoms1, atoms2):
            coords_1 = atoms1[atom_ids]
            coords_2 = atoms2[atom_ids]
            return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / len(coords_1))
        
        exp_atoms = np.array([a.coord for a in experimental_receptor.get_atoms()])
        comp_atoms = np.array([a.coord for a in computational_receptor.get_atoms()])
        rec_atoms_for_pocket = torch.tensor(
                np.array([a.coord for a in experimental_receptor.get_atoms() if a.name == 'CA']))
        ligand_noh = Chem.RemoveHs(copy.deepcopy(holo_ligand))
        ligand_noh_pos = ligand_noh.GetConformer().GetPositions().astype(np.float32)
        # Notice: the ligand we use here is holo while diffdock-pocket use the rdkit sampled conformer aligned with holo
        pocket_center, pocket_radius = _calculate_binding_pocket(rec_atoms_for_pocket, torch.from_numpy(ligand_noh_pos), 0)
        pocket_radius += 10
        idxs = np.array([pocket_selector(a.parent, pocket_center, pocket_radius) for a in computational_receptor.get_atoms()])
        if align_mistake == "exp_larger":
            rmsd = RMSD(idxs, comp_atoms, exp_atoms[:-1])
        elif align_mistake == "comp_larger":
            rmsd = RMSD(idxs[:-1], comp_atoms[:-1], exp_atoms)
        else:
            rmsd = RMSD(idxs, comp_atoms, exp_atoms)
        if istest != "test_apo" and istest != "test_holo":
            if rmsd > 2: # match_max_rmsd
                pocket_name = input_protein[0]
                receptor = experimental_receptor # use holo
            else:
                pocket_name = input_protein[1]
                receptor = computational_receptor # use apo
        elif istest == "test_apo":
            pocket_name = input_protein[1]
            receptor = computational_receptor
        else:
            pocket_name = input_protein[0]
            receptor = experimental_receptor
        
        ligand_h_pos = read_molecule(input_ligand, sanitize=True).GetConformer().GetPositions().astype(np.float32)
        ligand_center = np.mean(ligand_h_pos, axis=0)
        ligand_sphere_radius = np.sqrt(np.sum(np.square(ligand_h_pos - ligand_center), axis=1)).mean() # mean to max?
        pocket_radius = ligand_sphere_radius + 8 # buffer = 8
        config = {
            "pocket_center": ligand_center,
            "pocket_radius": pocket_radius,
        }
        residue_ids = []
        distance_lst = []
        atoms_num_list = []
        for i, chain in enumerate(receptor):
            for res_idx, residue in enumerate(chain):
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                        # distance of c_alpha to ligand center
                        distance = np.linalg.norm(c_alpha - ligand_center)
                        if distance < pocket_radius:
                            residue_ids.append(f'{chain.get_id()}_{residue.get_id()[1]}{residue.get_id()[2]}'.strip())
                            distance_lst.append(distance)
                            atoms_num_list.append(len(residue))
                            break
        
        # if pocket atom list is bigger than threshold drop residues!
        all_atom_num = sum(atoms_num_list)
        if all_atom_num > threshold:
            distance_lst = np.array(distance_lst)
            residue_ids = np.array(residue_ids)
            atoms_num_list = np.array(atoms_num_list)
            # sort the residues by distance to ligand center, erase the farthest residues until the threshold is reached
            sorted_idx = np.argsort(distance_lst)
            sorted_residue_ids = residue_ids[sorted_idx]
            sorted_atoms_num_list = atoms_num_list[sorted_idx]
            for i in reversed(range(len(sorted_residue_ids))):
                if all_atom_num < threshold:
                    break
                all_atom_num -= sorted_atoms_num_list[i]
            sorted_residue_ids = sorted_residue_ids[:i]
        else:
            sorted_residue_ids = residue_ids
        
        complex_name = complex_name if add_feature else None
        pocket_info_dict = self.process_pocket(receptor, experimental_receptor, sorted_residue_ids, pocket_name, complex_name)
        
        ret = {
            "atoms": ligand_atoms_list,
            "coordinates": coordinate_list,
            "mol_list": mol_list,
            "pocket_atoms": pocket_info_dict["atoms"], 
            "pocket_coordinates": pocket_info_dict["coords"],
            "restype": pocket_info_dict["restypes"],
            "restype_tokens": pocket_info_dict["restype_tokens"],
            # "side": side,
            "residue": pocket_info_dict["resnums"],
            "config": config,
            "holo_coordinates": [ligand_pos],
            "holo_mol": holo_ligand,
            "holo_pocket_coordinates": pocket_info_dict["holo_coords"],
            "smi": smiles,
            "pocket": pocket_name,
        }
        if add_feature:
            ret["atom_features_list"] = atom_features_list
            ret["bond_type_matrix"] = bond_type_matrix
            ret["lm_embeddings"] = pocket_info_dict["lm_embeddings"]
        return pickle.dumps(ret, protocol=-1,)
    
    def read_esm_embeddings(self, split, esm_embeddings_dir, complex_name_list, protein_file_list, start, end):
        if split in ["train", "valid", "test"]:
            if split == "valid": split = "val"
            esm_embeddings_path = os.path.join(esm_embeddings_dir, f"esm_embeddings_{split}.pt")
            if os.path.exists(esm_embeddings_path):
                esm_embeddings_list = torch.load(esm_embeddings_path)
                assert len(complex_name_list) == len(esm_embeddings_list), f"{len(esm_embeddings_list)} {len(complex_name_list)}"
                self.esm_embeddings = {}
                for complex_name, esm_emb in zip(complex_name_list[start:end], esm_embeddings_list[start:end]):
                    self.esm_embeddings[complex_name] = []
                    for chain in esm_emb:
                        self.esm_embeddings[complex_name].append(chain.cpu().numpy())
            else:
                raise NotImplementedError
                self.esm_embeddings = esm_utils.esm_embeddings_from_complexes(complex_name_list,
                                                                        protein_file_list,
                                                                        device=device)
                torch.save(esm_embeddings, esm_embeddings_path)
        else:
            raise NotImplementedError