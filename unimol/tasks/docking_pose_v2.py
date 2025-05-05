# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import contextlib
from typing import Optional
from collections.abc import Iterable

import numpy as np
import tqdm

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)
from unimol.data import (
    KeyDataset,
    LMDBDataset,
    ConformerSampleDockingPoseDataset,
    RightPadDatasetCoord,
    RightPadDatasetLAS,
    DistanceDataset,
    EdgeTypeDataset,
    CrossDistanceDataset,
    NormalizeDataset,
    NormalizeDockingPoseDataset,
    TTADockingPoseDataset,
    RightPadDatasetCross2D,
    CroppingPocketDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenPocketDataset,
    ReAlignLigandDataset,
    CustomNestedDictionaryDataset
)
from unicore import checkpoint_utils
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)

@register_task("docking_pose_v2")
class DockingPoseV2(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--finetune-model",
            default=None,
            type=str,
            help="pretrained whole model path",
        )
        parser.add_argument(
            "--bindnet-model",
            default=None,
            type=str,
            help="pretrained bindnet model path",
        )
        parser.add_argument(
            "--pocket-dict",
            default="dict_pkt.txt",
            type=str,
            help="pretrained bindnet model path",
        )
        parser.add_argument(
            "--use-small-dataset",
            default=False,
            type=bool,
            help="if use half size dataset",
        )
        parser.add_argument(
            "--use-flexible-docking",
            default="rigid",
            choices=["rigid", "flex_sc", "flex_all", "base_flex_sc", "base_flex_all"],
            type=str,
            help="if use half size dataset",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help='number of conformers generated with each molecule'
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=510,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--limit_dataset_size",
            type=int,
            default=-1,
            help="limit the dataset size",
        )
        
        parser.add_argument(
            "--limit_dataset_val_size",
            type=int,
            default=-1,
            help="limit the dataset size",
        )
        parser.add_argument(
            "--pocket_loss_weight",
            type=float,
            default=1.0,
            help="weight for pocket loss",
        )
        
        parser.add_argument(
            "--sc_aug",
            type=int,
            default=1,
            help="do the sidechain augmentation",
        )
        parser.add_argument(
            "--freeze_encoder",
            type=int,
            default=0,
            help="freeze the mol and pocket encoder",
        )
        parser.add_argument(
            "--add_feature",
            type=int,
            default=0,
            help="add input feature to coord decoder, including: atom_features_list, bond_type_matrix, lm_embeddings, restype_tokens",
        )

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.use_flexible_docking = self.args.use_flexible_docking
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, args.pocket_dict))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)
    
    def clear_invalid_sample(self, dataset):
        valid_indices = []
        for i in tqdm.tqdm(range(len(dataset)), "Clearing invalid samples"):
            try:
                _ = dataset[i]
                valid_indices.append(i)
            except:
                print("clear invalid", i)
                pass
        return valid_indices

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold', 'residue',
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + '.lmdb')
        
        if split == 'train' and self.args.limit_dataset_size > 0:
            dataset = LMDBDataset(data_path, self.args.use_small_dataset, self.args.limit_dataset_size)
        elif split=='valid' and self.args.limit_dataset_val_size > 0:
            dataset = LMDBDataset(data_path, self.args.use_small_dataset, self.args.limit_dataset_val_size)
        else:
            dataset = LMDBDataset(data_path, self.args.use_small_dataset)
        if split.startswith('train'):
            tgt_dataset = KeyDataset(dataset, 'target')
            smi_dataset = KeyDataset(dataset, 'smi')
            poc_dataset = KeyDataset(dataset, 'pocket')
            dataset = ConformerSampleDockingPoseDataset(dataset, self.args.seed, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', 'residue', True, self.use_flexible_docking, len(self.pocket_dictionary), self.args.sc_aug, self.args.add_feature)
        else:
            dataset = TTADockingPoseDataset(dataset, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', 'residue', True, self.args.conf_size, self.use_flexible_docking, len(self.pocket_dictionary), self.args.sc_aug, self.args.add_feature)
            tgt_dataset = KeyDataset(dataset, 'target')
            smi_dataset = KeyDataset(dataset, 'smi')
            poc_dataset = KeyDataset(dataset, 'pocket')

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)
        
        if self.args.add_feature:
            dataset = RemoveHydrogenPocketDataset(dataset, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', True, True, add_1dfeature_list=["lm_embeddings", "restype_tokens"],)
            dataset = CroppingPocketDataset(dataset, self.seed, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', "masked_tokens", self.args.max_pocket_atoms, ["lm_embeddings", "restype_tokens"])
            dataset = RemoveHydrogenPocketDataset(dataset, 'atoms', 'coordinates', 'holo_coordinates',  True, True, add_1dfeature_list=["atom_features_list"], add_2dfeature_list=["bond_type_matrix"])
        else:
            dataset = RemoveHydrogenPocketDataset(dataset, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', True, True)
            dataset = CroppingPocketDataset(dataset, self.seed, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', "masked_tokens", self.args.max_pocket_atoms)
            dataset = RemoveHydrogenPocketDataset(dataset, 'atoms', 'coordinates', 'holo_coordinates', True, True)
        
        apo_dataset = NormalizeDataset(dataset, 'coordinates')
        apo_dataset = NormalizeDataset(apo_dataset, 'pocket_coordinates')
        apo_dataset = ReAlignLigandDataset(dataset,'coordinates','pocket_coordinates')

        las_contraint = KeyDataset(apo_dataset, 'compound_LAS_edge_index')
        # las_contraint = FromNumpyDataset(las_contraint)


        src_dataset = KeyDataset(apo_dataset, 'atoms')
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        coord_dataset = KeyDataset(apo_dataset, 'coordinates')
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        if self.args.add_feature:
            src_atom_features_dataset = KeyDataset(apo_dataset, 'atom_features_list')
            src_atom_features_dataset = PrependAndAppend(src_atom_features_dataset, -1, -1) # 0 is valid value for features
            src_bond_type_dataset = KeyDataset(apo_dataset, 'bond_type_matrix')
            src_bond_type_dataset = PrependAndAppend2DDataset(src_bond_type_dataset, -1)

        src_pocket_dataset = KeyDataset(apo_dataset, 'pocket_atoms')
        src_pocket_dataset = TokenizeDataset(src_pocket_dataset, self.pocket_dictionary, max_seq_len=self.args.max_seq_len)
        coord_pocket_dataset = KeyDataset(apo_dataset, 'pocket_coordinates')
        src_pocket_dataset = PrependAndAppend(src_pocket_dataset, self.pocket_dictionary.bos(), self.pocket_dictionary.eos())
        pocket_edge_type = EdgeTypeDataset(src_pocket_dataset, len(self.pocket_dictionary))
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(distance_pocket_dataset, 0.0)
        src_pocket_masked_tokens_dataset = KeyDataset(apo_dataset, 'masked_tokens')
        src_pocket_masked_tokens_dataset = FromNumpyDataset(src_pocket_masked_tokens_dataset)
        src_pocket_masked_tokens_dataset = PrependAndAppend(src_pocket_masked_tokens_dataset, True, True)
        if self.args.add_feature:
            src_pocket_restype_dataset = KeyDataset(apo_dataset, 'restype_tokens')
            src_pocket_restype_dataset = PrependAndAppend(src_pocket_restype_dataset, 37, 37)
            src_pocket_lm_embedding_dataset = KeyDataset(apo_dataset, 'lm_embeddings')
            src_pocket_lm_embedding_dataset = PrependAndAppend(src_pocket_lm_embedding_dataset, 0.0, 0.0)

        holo_dataset = NormalizeDockingPoseDataset(dataset, 'holo_coordinates', 'holo_pocket_coordinates', 'holo_center_coordinates')
        holo_coord_dataset = KeyDataset(holo_dataset, 'holo_coordinates')
        holo_coord_dataset = FromNumpyDataset(holo_coord_dataset)
        holo_coord_pocket_dataset = KeyDataset(holo_dataset, 'holo_pocket_coordinates')
        holo_coord_pocket_dataset = FromNumpyDataset(holo_coord_pocket_dataset)

        holo_cross_distance_dataset = CrossDistanceDataset(holo_coord_dataset, holo_coord_pocket_dataset)

        holo_distance_dataset = DistanceDataset(holo_coord_dataset)
        holo_pocket_distance_dataset = DistanceDataset(holo_coord_pocket_dataset)
        holo_coord_dataset = PrependAndAppend(holo_coord_dataset, 0.0, 0.0)
        holo_distance_dataset = PrependAndAppend2DDataset(holo_distance_dataset, 0.0)
        holo_pocket_distance_dataset = PrependAndAppend2DDataset(holo_pocket_distance_dataset, 0.0)
        holo_coord_pocket_dataset = PrependAndAppend(holo_coord_pocket_dataset, 0.0, 0.0)
        holo_cross_distance_dataset = PrependAndAppend2DDataset(holo_cross_distance_dataset, 0.0)

        holo_center_coordinates = KeyDataset(holo_dataset, 'holo_center_coordinates')
        holo_center_coordinates = FromNumpyDataset(holo_center_coordinates)

        nest = {
                    "net_input": {
                        "mol_src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "mol_src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                        "pocket_src_tokens": RightPadDataset(
                            src_pocket_dataset,
                            pad_idx=self.pocket_dictionary.pad(),
                        ),
                        "pocket_src_coord": RightPadDatasetCoord(
                            coord_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_distance": RightPadDataset2D(
                            distance_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_edge_type": RightPadDataset2D(
                            pocket_edge_type,
                            pad_idx=0,
                        ),
                        "masked_tokens": RightPadDataset(
                            src_pocket_masked_tokens_dataset,
                            pad_idx=True,
                        ),
                        "las_constraint": RightPadDatasetLAS(
                            las_contraint,
                            pad_idx=0,
                        ),
                    },
                    "target": {
                        "distance_target": RightPadDatasetCross2D(holo_cross_distance_dataset, pad_idx=0),
                        "holo_coord": RightPadDatasetCoord(holo_coord_dataset, pad_idx=0),
                        "holo_coord_pocket": RightPadDatasetCoord(holo_coord_pocket_dataset, pad_idx=0),
                        "holo_distance_target": RightPadDataset2D(holo_distance_dataset, pad_idx=0),
                        "holo_pocket_distance_target": RightPadDataset2D(holo_pocket_distance_dataset, pad_idx=0),
                    },
                    "smi_name": RawArrayDataset(
                        smi_dataset
                    ),
                    "pocket_name": RawArrayDataset(
                        poc_dataset
                    ),
                    "holo_coord": RightPadDatasetCoord(
                        holo_coord_dataset,
                        pad_idx=0,
                    ),
                    "holo_coord_pocket": RightPadDatasetCoord(
                        holo_coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "holo_center_coordinates": RightPadDataset(
                        holo_center_coordinates,
                        pad_idx=0,
                    ),
                }
        if self.args.add_feature:
            pass
            nest["net_input"]["mol_atom_features"] = RightPadDatasetCross2D(src_atom_features_dataset, pad_idx=-1)
            nest["net_input"]["mol_bond_type"] = RightPadDatasetCross2D(src_bond_type_dataset, pad_idx=-1)
            nest["net_input"]["pocket_restype_tokens"] = RightPadDataset(src_pocket_restype_dataset, pad_idx=37)
            nest["net_input"]["pocket_lm_embeddings"] = RightPadDatasetCross2D(src_pocket_lm_embedding_dataset, pad_idx=0)
        nest_dataset = CustomNestedDictionaryDataset(nest)
        # nest_dataset.valid_indices = self.clear_invalid_sample(nest_dataset)
        # print("DEBUG: length", len(nest_dataset))
        if split.startswith('train'):
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset
    
    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.finetune_model is not None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_model, 
            )
            missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)
            print("missing_keys", missing_keys)
            print("unexpected_keys", unexpected_keys)
        elif args.bindnet_model is not None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.bindnet_model, 
            )
            
            def load_bindnet_weights(bindnet_key):
                bindnet_weights = {k: v for k, v in state["model"].items() if k.startswith(bindnet_key)}
                unimol_weights = {f'{k[len(bindnet_key):]}': v
                    for k, v in bindnet_weights.items()}
                return unimol_weights
            
            model.mol_model.embed_tokens.load_state_dict(load_bindnet_weights("lig_embed_tokens."))
            model.mol_model.gbf.load_state_dict(load_bindnet_weights("lig_gbf."))
            model.mol_model.gbf_proj.load_state_dict(load_bindnet_weights("lig_gbf_proj."))
            model.mol_model.encoder.load_state_dict(load_bindnet_weights("lig_encoder."))
            model.pocket_model.embed_tokens.load_state_dict(load_bindnet_weights("embed_tokens."))
            model.pocket_model.gbf.load_state_dict(load_bindnet_weights("gbf."))
            model.pocket_model.gbf_proj.load_state_dict(load_bindnet_weights("gbf_proj."))
            model.pocket_model.encoder.load_state_dict(load_bindnet_weights("encoder."))
            model.concat_decoder.load_state_dict(load_bindnet_weights("concat_decoder."))
            model.cross_distance_project.load_state_dict(load_bindnet_weights("cross_distance_project."))
            model.holo_distance_project.load_state_dict(load_bindnet_weights("holo_distance_project."))
        else:
            if args.finetune_mol_model is not None:
                print("load pretrain model weight from...", args.finetune_mol_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_mol_model, 
                )
                missing_keys, unexpected_keys = model.mol_model.load_state_dict(state["model"], strict=False)
                print("missing_keys", missing_keys)
                print("unexpected_keys", unexpected_keys)
            if args.finetune_pocket_model is not None:
                print("load pretrain model weight from...", args.finetune_pocket_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_pocket_model, 
                )
                missing_keys, unexpected_keys = model.pocket_model.load_state_dict(state["model"], strict=False)
                print("missing_keys", missing_keys)
                print("unexpected_keys", unexpected_keys)
            
        if self.args.freeze_encoder:
            model.mol_model.eval()
            model.pocket_model.eval()

        return model
