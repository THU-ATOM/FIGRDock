# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import torch
import torch.nn.functional as F
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import data_utils
from .unimol import UniMolModel, base_architecture, NonLinearHead, DistanceHead, GaussianLayer
from .transformer_encoder_with_pair import TransformerEncoderWithPair
import numpy as np
import os
import sys
interface_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'interface')
sys.path.append(interface_dir)
from pdbbind_benchmark_sc_utils import lig_feature_dims
from torch_scatter import scatter_add

logger = logging.getLogger(__name__)
from typing import Callable, Optional, Tuple, List
import torch as th

def kabsch(
    x: th.Tensor,
    y: th.Tensor,
    weights: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Kabsch alignment of X into Y (solution to least squares of point cloud rototranslation).
    Assumes X,Y are both ((...), N, D) - usually ((...), N, 3)
    Inputs:
    * x: ((...), N, D) th.Tensor
    * y: ((...), N, D) th.Tensor
    * weights: (..., N) th.Tensor. Optional. Only 0s and 1s to keep the algo meaningful
    Outputs:
    * x_: (..., N, D) th.Tensor
    """
    # (create and) ensure weights tensor is same shape as point clouds
    if weights is None:
        weights = th.ones_like(x[..., 0])

    try:
        # (..., n) -> (..., n, 1)
        weights = (weights / weights.sum(dim=-1, keepdim=True))[..., None]
        # calculate COM (...nd, ...n -> ... () d) and center
        x_mean = (x * weights).sum(dim=-2, keepdim=True)
        y_mean = (y * weights).sum(dim=-2, keepdim=True)
        x_ = x - x_mean
        y_ = y - y_mean

        # Optimal rotation matrix via SVD of covariance matrix (..., 3, 3)
        C = th.einsum("... n i, ... n j -> ... i j", y_ * weights, x_)
        U, S, V = th.linalg.svd(C)
        # Flip the sign of bottom row of each matrix if det product < 0
        det = th.det(V) * th.det(U)
        U_flip = th.ones_like(U)
        if det < 0:
            U_flip[:, -1] = -1.0
        U = U * U_flip

        # beware! th.linalg.svd(C).V.t() == th.svd(C).V
        R = U @ V
        # Note: R @ x == x @ R^(-1), and R^(-1) == Rt
        rt_x = th.einsum('...rc, ...nc -> ...nr', R, x_) + y_mean
    except Exception as e:
        rt_x = x_ + y_mean
    return rt_x

@register_model("docking_pose_v2")
class DockingPoseV2Model(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
        parser.add_argument(
            "--feat_position",
            type=str,
            default="None",
            choices=["None", "front", "back"],
            help="recycling nums of decoder",
        )
        
        # interpolation between the dock coord and init coord
        parser.add_argument(
            "--dock_interpolation",
            type=int,
            default=0,
            help="interpolation between the dock coord and init coord",
        )
        parser.add_argument(
            "--beta_dist_alpha",
            type=float,
            default=1.0,
            help="weight of pocket loss",
        )
        parser.add_argument(
            "--beta_dist_beta",
            type=float,
            default=10.0,
            help="weight of pocket loss",
        ) 
        
        # parser add the layer number of each module
        parser.add_argument(
            "--coord_decode_layers",
            type=int,
            default=4,
            help="number of encoder layers",
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
            "--las_init_opti",
            type=int,
            default=0,
            help="LAS optimization at the gradient of init coords in dock_with_gradient",
        )
        
        parser.add_argument(
            "--geometry_reg_step_size",
            type=float,
            default=0.001,
            help="LAS constraint steps",
        )
        
        parser.add_argument(
            "--split_encoder",
            type=int,
            default=0,
            help="the split encoder for the mol and pocket",
        )
        
    def __init__(self, args, mol_dictionary, pocket_dictionary, use_flexible_docking):
        super().__init__()
        unimol_docking_architecture(args)

        self.args = args
        self.geometry_reg_step_size = args.geometry_reg_step_size
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.mol_dictionary = mol_dictionary
        self.pocket_dictionary = pocket_dictionary
        self.use_flexible_docking = use_flexible_docking
        if len(pocket_dictionary) == 42:
            assert self.use_flexible_docking in ["rigid", "flex_sc", "flex_all"]
        self.concat_decoder = TransformerEncoderWithPair(
            encoder_layers=4,
            embed_dim=args.mol.encoder_embed_dim,
            ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
            attention_heads=args.mol.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, 'relu'
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, 'relu'
        )
        self.holo_pocket_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, 'relu'
        )

        K = 128
        dict_size = len(mol_dictionary) + len(pocket_dictionary)
        n_edge_type = dict_size * dict_size
        self.concat_gbf = GaussianLayer(K, n_edge_type)
        self.concat_gbf_proj = NonLinearHead(
            K, args.mol.encoder_attention_heads, args.mol.activation_fn
        )

        # self.mol_atom_embedding_list = torch.nn.ModuleList()
        # self.num_categorical_features = len(lig_feature_dims[0])
        # for _, dim in enumerate(lig_feature_dims[0]):
        #     emb = torch.nn.Embedding(dim, args.mol.encoder_embed_dim, -1)
        #     torch.nn.init.xavier_uniform_(emb.weight.data)
        #     self.mol_atom_embedding_list.append(emb)
        
        # emb = torch.nn.Embedding(38, 60, -1)
        # torch.nn.init.xavier_uniform_(emb.weight.data)
        # self.pocket_embedding_list = torch.nn.ModuleList([emb])

        # self.pocket_embedding_proj = NonLinearHead(
        #     1280, args.mol.encoder_embed_dim, args.mol.activation_fn
        # )

        # self.mol_bond_embedding_list = torch.nn.ModuleList()
        # emb = torch.nn.Embedding(5, args.mol.encoder_attention_heads, -1)
        # torch.nn.init.xavier_uniform_(emb.weight.data)
        # self.mol_bond_embedding_list.append(emb)

        # self.mol_rep_proj = NonLinearHead(
        #     args.mol.encoder_embed_dim+60, args.mol.encoder_embed_dim, args.mol.activation_fn
        # )
        self.pocket_rep_proj = NonLinearHead(
            args.mol.encoder_embed_dim+1280, args.mol.encoder_embed_dim, args.mol.activation_fn
        )
        # self.mol_pair_rep_proj = NonLinearHead(
        #     args.mol.encoder_attention_heads*2, args.mol.encoder_attention_heads, args.mol.activation_fn
        # )

        self.coord_decoder = TransformerEncoderWithPair(
            encoder_layers=args.coord_decode_layers,
            embed_dim=args.mol.encoder_embed_dim,
            ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
            attention_heads=args.mol.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        
        if self.args.split_encoder:
            self.pocket_coord_decoder = TransformerEncoderWithPair(
                encoder_layers=args.coord_decode_layers,
                embed_dim=args.mol.encoder_embed_dim,
                ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
                attention_heads=args.mol.encoder_attention_heads,
                emb_dropout=0.1,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                activation_fn="gelu",
            )
        if self.args.feat_position == "back":
            self.pocket_rep_proj = NonLinearHead(
                args.mol.encoder_embed_dim+1280, args.mol.encoder_embed_dim, args.mol.activation_fn
                )
        
        
        self.coord_delta_project = NonLinearHead(
            args.mol.encoder_attention_heads, 1, args.mol.activation_fn
        )
        self.pocket_coord_delta_project = NonLinearHead(
            args.mol.encoder_attention_heads, 1, args.mol.activation_fn
        )
        self.prmsd_project = NonLinearHead(
            args.mol.encoder_embed_dim, 32, args.mol.activation_fn
        )
        self.pocket_prmsd_project = NonLinearHead(
            args.mol.encoder_embed_dim, 32, args.mol.activation_fn
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary, task.use_flexible_docking)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_coord,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_coord,
        pocket_src_edge_type,
        masked_tokens=None,
        features_only=True,
        mol_atom_features=None,
        mol_bond_type=None,
        pocket_restype_tokens=None,
        pocket_lm_embeddings=None,
        las_constraint=None,
        **kwargs
    ):
        def get_dist_features(dist, et, flag):
            if flag == 'mol':
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'pocket':
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'concat':
                n_node = dist.size(-1)
                gbf_feature = self.concat_gbf(dist, et)
                gbf_result = self.concat_gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                return None

        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_atom_mask = mol_src_tokens > 2
        pocket_atom_mask = pocket_src_tokens > 2
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(mol_src_distance, mol_src_edge_type, 'mol')
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_encoder_rep = mol_outputs[0]
        mol_encoder_pair_rep = mol_outputs[1]

        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = get_dist_features(pocket_src_distance, pocket_src_edge_type, 'pocket')
        pocket_outputs = self.pocket_model.encoder(pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias)
        pocket_encoder_rep = pocket_outputs[0]
        pocket_encoder_pair_rep = pocket_outputs[1]

        mol_sz = mol_encoder_rep.size(1)
        pocket_sz = pocket_encoder_rep.size(1)
        cross_distance_mask, distance_mask, coord_mask = calc_mask(mol_atom_mask, pocket_atom_mask)

        concat_rep = torch.cat([mol_encoder_rep, pocket_encoder_rep], dim=-2) # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat([mol_padding_mask, pocket_padding_mask], dim=-1)   # [batch, mol_sz+pocket_sz]
        attn_bs = mol_graph_attn_bias.size(0)

        concat_attn_bias = torch.zeros(attn_bs, mol_sz+pocket_sz, mol_sz+pocket_sz).type_as(concat_rep)  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]
        concat_attn_bias[:,:mol_sz,:mol_sz] = mol_encoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, mol_sz, mol_sz).contiguous()
        concat_attn_bias[:,-pocket_sz:,-pocket_sz:] = pocket_encoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, pocket_sz, pocket_sz).contiguous()

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep)
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i!=(self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, mol_sz+pocket_sz, mol_sz+pocket_sz)

        decoder_rep = decoder_outputs[0]
        decoder_pair_rep = decoder_outputs[1]

        mol_decoder = decoder_rep[:,:mol_sz]
        pocket_decoder = decoder_rep[:,mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:,:mol_sz,:mol_sz,:]
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:,:mol_sz,mol_sz:,:] + decoder_pair_rep[:,mol_sz:,:mol_sz,:].transpose(1,2))/2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float('-inf')] = 0

        cross_rep = torch.cat([
                                mol_pocket_pair_decoder_rep,
                                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1), 
                                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1), 
                                ], dim=-1)   # [batch, mol_sz, pocket_sz, 4*hidden_size]

        
        
        cross_distance_predict = F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat([
                                mol_pair_decoder_rep,
                                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1), 
                                ], dim=-1) # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep)  # batch, mol_sz, mol_sz
        if self.use_flexible_docking != "rigid":
            pocket_pair_decoder_rep = decoder_pair_rep[:,mol_sz:,mol_sz:,:]
            holo_pocket_encoder_pair_rep = torch.cat([
                                pocket_pair_decoder_rep,
                                pocket_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1), 
                                ], dim=-1) # [batch, pocket_sz, pocket_sz, 3*hidden_size]
            holo_pocket_distance_predict = self.holo_pocket_distance_project(holo_pocket_encoder_pair_rep) # batch, pocket_sz, pocket_sz

        if self.args.las_init_opti > 0:
            mol_src_coord_update = dock_with_gradient(mol_src_coord, pocket_src_coord, cross_distance_predict, holo_distance_predict, cross_distance_mask, distance_mask, las_constraint=las_constraint)
        else:        
            mol_src_coord_update = dock_with_gradient(mol_src_coord, pocket_src_coord, cross_distance_predict, holo_distance_predict, cross_distance_mask, distance_mask)

        if self.args.dock_interpolation > 0:
            sample_num = mol_src_coord.size(0)
            alpha = self.args.beta_dist_alpha
            beta = self.args.beta_dist_beta
            weight = np.random.beta(alpha, beta, sample_num)
            weight = torch.tensor(weight).type_as(mol_src_coord).unsqueeze(-1).unsqueeze(-1)
            mol_src_coord_update = (1 - weight) * mol_src_coord_update + weight * mol_src_coord

        num_types = len(self.mol_dictionary) + len(self.pocket_dictionary)
        node_input = torch.concat([mol_src_tokens, pocket_src_tokens + len(self.mol_dictionary)], dim=1) # [batch, mol_sz+pocket_sz]
        concat_edge_type = node_input.unsqueeze(-1) * num_types + node_input.unsqueeze(-2)
        
        if mol_atom_features is not None and self.args.feat_position == "back":
            mol_decoder_rep_feat = decoder_rep[:,:mol_sz]
            # for i in range(self.num_categorical_features):
            #     mol_atom_feat = mol_atom_features[:,:,i].clone()
            #     mol_atom_feat[mol_atom_feat == -1] = lig_feature_dims[0][i] - 1
            #     if mol_decoder_rep_feat is None:
            #         mol_decoder_rep_feat = self.mol_atom_embedding_list[i](mol_atom_feat).unsqueeze(-1)
            #     else:
            #         mol_decoder_rep_feat = torch.cat((mol_decoder_rep_feat, self.mol_atom_embedding_list[i](mol_atom_feat).unsqueeze(-1)), dim=-1)
            # mol_decoder_rep_feat = torch.mean(mol_decoder_rep_feat, dim=-1)
            # mol_decoder_rep_feat = torch.cat((decoder_rep[:,:mol_sz], mol_decoder_rep_feat), dim=-1)
            # mol_decoder_rep_feat = self.mol_rep_proj(mol_decoder_rep_feat)

            # pocket_decoder_rep_feat = self.pocket_embedding_list[0](pocket_restype_tokens)
            pocket_decoder_rep_feat = torch.cat((decoder_rep[:,mol_sz:], pocket_lm_embeddings), dim=-1) # pocket_lm_embeddings
            pocket_decoder_rep_feat = self.pocket_rep_proj(pocket_decoder_rep_feat)

            decoder_rep_for_pocket = torch.cat((mol_decoder_rep_feat, pocket_decoder_rep_feat), dim=1)
        else:
            decoder_rep_for_pocket = decoder_rep
            
         
        def coord_decoder(mol_src_coord_update, pocket_src_coord_update, mol_src_coord, las_constraint=None, last_iter=False):
            concat_coord = torch.cat([mol_src_coord_update, pocket_src_coord_update], dim=1) # [batch, mol_sz+pocket_sz, 3]
            concat_distance = (concat_coord.unsqueeze(1) - concat_coord.unsqueeze(2)).norm(dim=-1)
            concat_attn_bias = get_dist_features(concat_distance, concat_edge_type, 'concat')
            if False and mol_bond_type is not None and self.args.feat_position == "back":
                mol_bond_type[mol_bond_type == -1] = 4
                mol_decoder_pair_rep_feat = self.mol_bond_embedding_list[0](mol_bond_type.view(-1, mol_sz*mol_sz))\
                        .view(-1, mol_sz, mol_sz)
                mol_decoder_pair_rep_feat = torch.cat((concat_attn_bias[:,:mol_sz,:mol_sz], mol_decoder_pair_rep_feat), dim=-1)
                mol_decoder_pair_rep_feat = mol_decoder_pair_rep_feat.view(-1, mol_sz, mol_sz, self.args.mol.encoder_attention_heads*2)
                mol_decoder_pair_rep_feat = self.mol_pair_rep_proj(mol_decoder_pair_rep_feat).view(-1, mol_sz, mol_sz)
                concat_attn_bias = torch.cat(
                    (torch.cat((mol_decoder_pair_rep_feat, concat_attn_bias[:,:mol_sz,mol_sz:]), dim=2),
                    concat_attn_bias[:,mol_sz:,:]), dim=1
                )
            concat_outputs = self.coord_decoder(decoder_rep, padding_mask=concat_mask, attn_mask=concat_attn_bias)
            coord_decoder_rep = concat_outputs[0]
            coord_decoder_rep = coord_decoder_rep[:,:mol_sz,:]
            delta_decoder_pair_rep = concat_outputs[2]   
            delta_decoder_rep = delta_decoder_pair_rep[:,:mol_sz,:mol_sz,:]

            atom_num = (torch.sum(~mol_padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            delta_pos = mol_src_coord_update.unsqueeze(1) - mol_src_coord_update.unsqueeze(2)
            attn_probs = self.coord_delta_project(delta_decoder_rep)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = torch.sum(coord_update, dim=2)
            mol_src_coord_update = mol_src_coord_update + coord_update #* 10
            
            
            if self.args.split_encoder:
                concat_outputs_for_pocket = self.pocket_coord_decoder(decoder_rep_for_pocket, padding_mask=concat_mask, attn_mask=concat_attn_bias)
                pocket_coord_decoder_rep = concat_outputs_for_pocket[0][:,mol_sz:,:]
            else:
                pocket_coord_decoder_rep = concat_outputs[0][:,mol_sz:,:]
            if self.use_flexible_docking != "rigid":
                if self.args.split_encoder:
                    pocket_delta_decoder_rep = concat_outputs_for_pocket[2][:,mol_sz:,mol_sz:,:]
                else:
                    pocket_delta_decoder_rep = delta_decoder_pair_rep[:,mol_sz:,mol_sz:,:]
                pocket_update_mask = ~torch.logical_or(masked_tokens, pocket_padding_mask)
                pocket_update_atom_num = (torch.sum(pocket_update_mask, dim=1) - 1).view(-1, 1, 1, 1)
                pocket_delta_pos = pocket_src_coord_update.unsqueeze(1) - pocket_src_coord_update.unsqueeze(2)
                pocket_attn_probs = self.pocket_coord_delta_project(pocket_delta_decoder_rep)
                pocket_coord_update = pocket_delta_pos / pocket_update_atom_num * pocket_attn_probs
                pocket_coord_update = torch.sum(pocket_coord_update, dim=2)
                pocket_src_coord_update = pocket_src_coord_update + pocket_coord_update.masked_fill_(~pocket_update_mask.unsqueeze(-1),0)
                
            
            if self.args.geom_reg_steps > 0 and last_iter:
                batch_size, mol_max_atom_num = mol_src_coord_update.size()[:2]
                las_edge_offset = torch.tensor([mol_max_atom_num * i for i in range(batch_size)]).type_as(mol_src_coord_update)
                las_edge_offset.unsqueeze_(1).unsqueeze_(1)
                
                las_edge_mask = (las_constraint.sum(dim=2) != 0)
                
                las_constraint_new = las_constraint + 1 # the zero index of coordinate is cls token(padding as zero)
                
                # reshape the mol_src_coord_update
                mol_src_coord_update_merge = mol_src_coord_update.view(-1, 3)
                mol_src_coord_merge = mol_src_coord.view(-1, 3)
                
                las_constraint_new = las_constraint_new + las_edge_offset
                
                las_constraint_valid = las_constraint_new[las_edge_mask].to(torch.long)
                
                # if las_constraint_valid.max() >= mol_src_coord_merge.shape[0]:
                #     print('Debug')
                #     import pdb; pdb.set_trace()
                
                assert las_constraint_valid.max() < mol_src_coord_merge.shape[0]
                assert las_constraint_valid.max() < mol_src_coord_update_merge.shape[0]
                
                coords = torch.ones_like(mol_src_coord_update).type_as(mol_src_coord_update) * mol_src_coord_update.detach()
                coords.requires_grad = True
                optimizer = torch.optim.LBFGS([coords], lr=self.geometry_reg_step_size)
                
                
                # for i in range(self.args.geom_reg_steps):
                #     # optimizer.zero_grad()
                #     def closure():
                #         optimizer.zero_grad()
                #         mol_src_coord_update_merge = coords.view(-1, 3) # the predicted structure
                #         mol_src_coord_merge = mol_src_coord.view(-1, 3) # the inference structure
                #         LAS_cur_squared = torch.sum((mol_src_coord_update_merge[las_constraint_valid[:, 0]] - mol_src_coord_update_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                #         LAS_true_squared = torch.sum((mol_src_coord_merge[las_constraint_valid[:, 0]] - mol_src_coord_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                #         las_loss = (LAS_cur_squared - LAS_true_squared).pow(2).mean()
                        
                        

                #         loss = las_loss 
                #         # + 
                #         coord_mask = mol_src_coord.ne(0)[:,:,0]
                #         # calculate the rmsd
                #         # rmsd_loss = torch.sqrt(torch.sum(((coords - mol_src_coord)[coord_mask]) ** 2, dim=1)).mean()
                        
                #         rmsd_loss = torch.sqrt(torch.sum(((coords - mol_src_coord_update)[coord_mask]) ** 2, dim=1)).mean()
                        
                #         # if rmsd_loss.item() > 0 and i > 100:
                #         #     # calculate the rmsd between coords and mol_src_coord_update(original coords)
                            
                #         #     loss += 0.1 * rmsd_loss
                        
                #         # print('iters: ', i)
                #         # print('las_loss:', las_loss.item())
                #         # print('rmsd_loss: ', rmsd_loss.item())
                        
                #         loss.backward(retain_graph=True)
                #         return loss
                    
                #     loss = optimizer.step(closure)
                
                # org_mol_src_coord_update = mol_src_coord_update.clone()
                # mol_src_coord_update = coords.detach()
                
                # batch_num = mol_src_coord_update.shape[0]
                # coord_mask = mol_src_coord.ne(0)[:,:,0]
                # rmsd_loss = torch.sqrt(torch.sum(((mol_src_coord_update - org_mol_src_coord_update)[coord_mask]) ** 2, dim=1)).mean()
                # print('before kabsh rmsd_loss: ', rmsd_loss.item())
                # for i in range(batch_num):
                #     mol_src_coord_update[i][coord_mask[i]] = kabsch(mol_src_coord_update[i][coord_mask[i]], org_mol_src_coord_update[i][coord_mask[i]])
                
                # rmsd_loss = torch.sqrt(torch.sum(((mol_src_coord_update - org_mol_src_coord_update)[coord_mask]) ** 2, dim=1)).mean()
                # print('after kbash rmsd_loss: ', rmsd_loss.item())
                
                
                
                for i in range(self.args.geom_reg_steps):
                    
                    
                    
                    LAS_cur_squared = torch.sum((mol_src_coord_update_merge[las_constraint_valid[:, 0]] - mol_src_coord_update_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                    LAS_true_squared = torch.sum((mol_src_coord_merge[las_constraint_valid[:, 0]] - mol_src_coord_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                    
                    las_loss = (LAS_cur_squared - LAS_true_squared).pow(2).mean()
                    # print('iteration is,', i)
                    # print('las loss is', las_loss.item())
                    
                    
                    
                    grad_squared = 2 * (mol_src_coord_update_merge[las_constraint_valid[:, 0]] - mol_src_coord_update_merge[las_constraint_valid[:, 1]])
                    LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                    LAS_delta_coord = scatter_add(src=LAS_force, index=las_constraint_valid[:, 1], dim=0, dim_size=mol_src_coord_update_merge.shape[0])
                    mol_src_coord_update_merge = mol_src_coord_update_merge + (LAS_delta_coord * self.geometry_reg_step_size).clamp(min=-15, max=15)
                    
                    
                    # mol_src_coord_update = dock_with_gradient(mol_src_coord_update, pocket_src_coord_update, cross_distance_predict, holo_distance_predict, cross_distance_mask, distance_mask)
                org_mol_src_coord_update = mol_src_coord_update.clone()
                mol_src_coord_update = mol_src_coord_update_merge.view(batch_size, mol_max_atom_num, 3)
                
                # batch_num = mol_src_coord_update.shape[0]
                # coord_mask = mol_src_coord.ne(0)[:,:,0]
                # rmsd_loss = torch.sqrt(torch.sum(((mol_src_coord_update - org_mol_src_coord_update)[coord_mask]) ** 2, dim=1)).mean()
                # print('before kabsh rmsd_loss: ', rmsd_loss.item())
                # for i in range(batch_num):
                #     mol_src_coord_update[i][coord_mask[i]] = kabsch(mol_src_coord_update[i][coord_mask[i]], org_mol_src_coord_update[i][coord_mask[i]])
                
                # rmsd_loss = torch.sqrt(torch.sum(((mol_src_coord_update - org_mol_src_coord_update)[coord_mask]) ** 2, dim=1)).mean()
                # print('after kbash rmsd_loss: ', rmsd_loss.item())
                
            
            return mol_src_coord_update, coord_decoder_rep, pocket_src_coord_update, pocket_coord_decoder_rep

        pocket_src_coord_update = pocket_src_coord
        if self.training:
            with data_utils.numpy_seed(self.get_num_updates()):
                recycling = np.random.randint(self.args.coord_decode_total_iter)
                for i in range(recycling):
                    with torch.no_grad():
                        if self.args.pocket_loss_weight == 0:
                            pocket_src_coord_update = pocket_src_coord
                        mol_src_coord_update, _, pocket_src_coord_update, _ = coord_decoder(mol_src_coord_update, pocket_src_coord_update, mol_src_coord,las_constraint)
            mol_src_coord_update, coord_decoder_rep, pocket_src_coord_update, pocket_coord_decoder_rep = coord_decoder(mol_src_coord_update, pocket_src_coord_update, mol_src_coord, las_constraint)
            # pass
            # coord_decoder_rep = mol_decoder
            # pocket_coord_decoder_rep = pocket_decoder
        else:
            # recycling = 4
            recycling = self.args.coord_decode_total_iter
            for i in range(recycling):
                if self.args.pocket_loss_weight == 0:
                    pocket_src_coord_update = pocket_src_coord
                # if i == 1:
                #     print('debug')
                    # import pdb; pdb.set_trace()
                # print('recycling:', i)
                # print('mol_src_coord_update shape: ', mol_src_coord_update.shape)
                # print('las_constraint shape: ', las_constraint.shape)
                last_iter = 0
                if i == recycling - 1:
                    last_iter = 1
                
                mol_src_coord_update, coord_decoder_rep, pocket_src_coord_update, pocket_coord_decoder_rep = coord_decoder(mol_src_coord_update, pocket_src_coord_update, mol_src_coord, las_constraint, last_iter=last_iter)

        if self.use_flexible_docking == "rigid":
            prmsd_predict = self.prmsd_project(coord_decoder_rep)
            return cross_distance_predict, holo_distance_predict, mol_src_coord_update, prmsd_predict
        else:
            # prmsd
            prmsd_predict = self.prmsd_project(coord_decoder_rep)
            pocket_prmsd_predict = self.pocket_prmsd_project(pocket_coord_decoder_rep)
            return cross_distance_predict, holo_distance_predict, holo_pocket_distance_predict, mol_src_coord_update, pocket_src_coord_update, prmsd_predict, pocket_prmsd_predict


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


def calc_mask(mol_padding_mask, pocket_padding_mask):
    mol_sz = mol_padding_mask.size()
    pocket_sz = pocket_padding_mask.size()
    cross_distance_mask = torch.zeros(mol_sz[0], mol_sz[1], pocket_sz[1]).type_as(mol_padding_mask)
    cross_distance_mask = mol_padding_mask.unsqueeze(-1) & pocket_padding_mask.unsqueeze(-2)
    distance_mask = torch.zeros(mol_sz[0], mol_sz[1], mol_sz[1]).type_as(mol_padding_mask)
    distance_mask = mol_padding_mask.unsqueeze(-1) & mol_padding_mask.unsqueeze(-2)
    coord_mask = torch.zeros(mol_sz[0], mol_sz[1], 3).type_as(mol_padding_mask)
    coord_mask.masked_fill_(
        mol_padding_mask.unsqueeze(-1),
        True
    )
    return cross_distance_mask, distance_mask, coord_mask


def scoring_function(predict_coords, pocket_coords, distance_predict, holo_distance_predict, cross_distance_mask, distance_mask, dist_threshold=4.5):
    dist = torch.norm(predict_coords.unsqueeze(-2) - pocket_coords.unsqueeze(-3), dim=-1)   # bs, mol_sz, pocket_sz
    holo_dist = torch.norm(predict_coords.unsqueeze(-2) - predict_coords.unsqueeze(-3), dim=-1) # bs, mol_sz, mol_sz

    cross_distance_mask = (distance_predict < dist_threshold) & cross_distance_mask
    cross_dist_score = ((dist[cross_distance_mask] - distance_predict[cross_distance_mask])**2).mean()
    dist_score = ((holo_dist[distance_mask] - holo_distance_predict[distance_mask])**2).mean()
    loss = cross_dist_score + dist_score
    return loss


def dock_with_gradient(mol_coords, pocket_coords, distance_predict, holo_distance_predict, cross_distance_mask, distance_mask, iterations=20, early_stoping=5, las_constraint=None):

    coords = torch.ones_like(mol_coords).type_as(mol_coords) * mol_coords
    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=1.0)
    bst_loss, times = 10000.0, 0
    distance_predict_detached = distance_predict.detach().clone()
    holo_distance_predict_detached = holo_distance_predict.detach().clone()
    
    if las_constraint is not None:
        # for LAS
        batch_size, mol_max_atom_num = mol_coords.size()[:2]
        las_edge_offset = torch.tensor([mol_max_atom_num * i for i in range(batch_size)]).type_as(mol_coords)
        las_edge_offset.unsqueeze_(1).unsqueeze_(1)
        
        las_edge_mask = (las_constraint.sum(dim=2) != 0)
        
        las_constraint_new = las_constraint + 1 # the zero index of coordinate is cls token(padding as zero)
        
        
        
        las_constraint_new = las_constraint_new + las_edge_offset
        
        las_constraint_valid = las_constraint_new[las_edge_mask].to(torch.long)
        
        # if las_constraint_valid.max() >= mol_src_coord_merge.shape[0]:
        #     print('Debug')
        #     import pdb; pdb.set_trace()
        
        # assert las_constraint_valid.max() < mol_src_coord_merge.shape[0]
        # assert las_constraint_valid.max() < mol_src_coord_update_merge.shape[0]
    
    
    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            loss = scoring_function(coords, pocket_coords, distance_predict_detached, holo_distance_predict_detached, cross_distance_mask, distance_mask)
            # print(f'org loss is : {loss}')
            if las_constraint is not None:
                # reshape the mol_src_coord_update
                mol_src_coord_update_merge = coords.view(-1, 3)
                mol_src_coord_merge = mol_coords.view(-1, 3)
                LAS_cur_squared = torch.sum((mol_src_coord_update_merge[las_constraint_valid[:, 0]] - mol_src_coord_update_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                LAS_true_squared = torch.sum((mol_src_coord_merge[las_constraint_valid[:, 0]] - mol_src_coord_merge[las_constraint_valid[:, 1]]) ** 2, dim=1)
                las_loss = (LAS_cur_squared - LAS_true_squared).pow(2).mean()
                
                
                # print(f'las_loss is: {las_loss}' )
                
                loss = loss + (LAS_cur_squared - LAS_true_squared).pow(2).mean()
                
                
                # grad_squared = 2 * (coords[las_constraint_valid[:, 0]] - coords[las_constraint_valid[:, 1]])
                # LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                # LAS_delta_coord = scatter_add(src=LAS_force, index=las_constraint_valid[:, 1], dim=0, dim_size=mol_src_coord_update_merge.shape[0])
                # coords = coords + (LAS_delta_coord * 0.001).clamp(min=-15, max=15)
            
            
            loss.backward(retain_graph=True)
            return loss
        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0 
        else:
            times += 1
            if times > early_stoping:
                break
    return coords.detach()


@register_model_architecture("docking_pose_v2", "docking_pose_v2")
def unimol_docking_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(args, "pocket_encoder_attention_heads", 64)
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(args, "pocket_pooler_activation_fn", "tanh")
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0
    
    base_architecture(args)