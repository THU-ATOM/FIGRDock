# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

@register_loss("flexible_docking_pose_v2")
class FlexibleDockingPosseV2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.padding_idx = task.dictionary.pad()
        self.args = task.args

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert model.use_flexible_docking != "rigid"
        # freeze the model parameters, if param name not contains 'prmsd_project'
        # rmsd_param_lst = []
        # for name, param in model.named_parameters():
        #     if 'prmsd_project' in name:
        #         param.requires_grad = True
        #         rmsd_param_lst.append(name)
        #     else:
        #         param.requires_grad = False
                
        # print(f'rmsd params lst is {rmsd_param_lst}')
        
        net_output = model(**sample["net_input"])
        
        cross_distance_predict, holo_distance_predict, holo_pocket_distance_predict, coord_predict, pocket_coord_predict, prmsd_predict, pocket_prmsd_predict = net_output[:7]

        ### distance loss
        distance_mask = sample["target"]["distance_target"].ne(0) # 0 is padding
        if self.args.dist_threshold > 0:
            distance_mask &= sample["target"]["distance_target"] < self.args.dist_threshold
        distance_predict = cross_distance_predict[distance_mask]
        distance_target =  sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), 
            distance_target.float(), 
            reduction="mean")
        
        ### holo distance loss
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict = holo_distance_predict[holo_distance_mask]
        holo_distance_target =  sample["target"]["holo_distance_target"][holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict.float(), 
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )
        
        torch.set_printoptions(profile="full")
        pocket_token_mask = sample["net_input"]["pocket_src_tokens"].ne(self.padding_idx) & \
                     sample["net_input"]["pocket_src_tokens"].ne(self.eos_idx) & \
                     sample["net_input"]["pocket_src_tokens"].ne(self.bos_idx) & \
                     ~sample["net_input"]["masked_tokens"]
        holo_pocket_distance_mask = pocket_token_mask.unsqueeze(-1) & pocket_token_mask.unsqueeze(1)
        holo_pocket_distance_predict = holo_pocket_distance_predict[holo_pocket_distance_mask]
        holo_pocket_distance_target =  sample["target"]["holo_pocket_distance_target"][holo_pocket_distance_mask]
        holo_pocket_distance_loss = F.smooth_l1_loss(
            holo_pocket_distance_predict.float(), 
            holo_pocket_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )
        

        ### coord loss
        coord_target = sample["target"]["holo_coord"]
        coord_mask = coord_target.ne(0)  # 0 is padding
        coord_loss = (((coord_predict - coord_target)**2).sum(dim=[1,2]) / coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()
        
        pocket_coord_target = sample["target"]["holo_coord_pocket"]
        pocket_coord_mask = pocket_coord_target.ne(0) & pocket_token_mask.unsqueeze(-1) # 0 is padding
        pocket_coord_loss = (((pocket_coord_predict - pocket_coord_target)**2).sum(dim=[1,2]) / pocket_coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()

        ### prmsd loss
        tick = 0.25
        max_bins = 32 
        token_mask = coord_mask[:,:,0]
        prmsd_target = ((coord_predict - coord_target)**2 * coord_mask).sum(dim=-1).sqrt()
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_predict = F.log_softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_loss = F.nll_loss(
            prmsd_predict[token_mask],
            prmsd_target[token_mask],
            reduction="mean",
        )
        
        pocket_token_mask = pocket_coord_mask[:,:,0]
        pocket_prmsd_target = ((pocket_coord_predict - pocket_coord_target)**2 * pocket_coord_mask).sum(dim=-1).sqrt()
        pocket_prmsd_target = (pocket_prmsd_target / tick).long()
        pocket_prmsd_target[pocket_prmsd_target >= (max_bins - 1)] = max_bins - 1
        pocket_prmsd_target[pocket_prmsd_target < 0] = 0
        pocket_prmsd_logit = F.softmax(pocket_prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        pocket_prmsd_predict = F.log_softmax(pocket_prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        pocket_prmsd_loss = F.nll_loss(
            pocket_prmsd_predict[pocket_token_mask],
            pocket_prmsd_target[pocket_token_mask],
            reduction="mean",
        )

        # print("DEBUG: distance loss", distance_loss, holo_distance_loss, holo_pocket_distance_loss)
        # print("DEBUG: coord loss", coord_loss, pocket_coord_loss)
        # print("DEBUG: prmsd loss", prmsd_loss*0.1, pocket_prmsd_loss * 0.1)
        loss = distance_loss + holo_distance_loss + prmsd_loss*0 + coord_loss + (pocket_coord_loss + holo_pocket_distance_loss + pocket_prmsd_loss * 0) * self.args.pocket_loss_weight
        # loss = distance_loss + holo_distance_loss + prmsd_loss*0.0 + coord_loss + (pocket_coord_loss + holo_pocket_distance_loss + pocket_prmsd_loss * 0.0) * self.args.pocket_loss_weight
        # loss = prmsd_loss + pocket_prmsd_loss

        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)
        pocket_weight = torch.arange(max_bins,).type_as(pocket_prmsd_logit).unsqueeze(0) + tick / 2
        pocket_prmsd_score = (pocket_prmsd_logit * pocket_weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_distance_loss": distance_loss.data,
            "distance_loss": holo_distance_loss.data,
            "pocket_distance_loss": holo_pocket_distance_loss.data, # New: pocket loss
            "coord_loss": coord_loss.data,
            "pocket_coord_loss": pocket_coord_loss.data, # New: pocket loss
            "prmsd_loss": prmsd_loss.data,
            "pocket_prmsd_loss": pocket_prmsd_loss.data, # New: pocket loss
            "prmsd_score": prmsd_score.data,
            "pocket_prmsd_score": pocket_prmsd_score.data, # New: pocket score
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,   # last iteration
            "coord_target": sample["target"]["holo_coord"].data,
            "pocket_coord_predict": pocket_coord_predict.data,   # last iteration
            "pocket_coord_target": sample["target"]["holo_coord_pocket"].data,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict.data.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.data.detach().cpu()
            logging_output["atoms"] = sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            logging_output["pocket_atoms"] = sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            logging_output["coordinates"] = sample["net_input"]["mol_src_coord"].data.detach().cpu()
            logging_output["holo_coordinates"] = sample["target"]["holo_coord"].data.detach().cpu()
            logging_output["pocket_coordinates"] = sample["net_input"]["pocket_src_coord"].data.detach().cpu()
            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()

        return loss, sample_size, logging_output
        

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=3
        )
        cross_distance_loss = sum(log.get("cross_distance_loss", 0) for log in logging_outputs)
        if cross_distance_loss > 0:
            metrics.log_scalar(
                "cross_distance_loss", cross_distance_loss / sample_size, sample_size, round=3
            )
        distance_loss = sum(log.get("distance_loss", 0) for log in logging_outputs)
        if distance_loss > 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / sample_size, sample_size, round=3
            )
        pocket_distance_loss = sum(log.get("pocket_distance_loss", 0) for log in logging_outputs)
        if pocket_distance_loss > 0:
            metrics.log_scalar(
                "pocekt_distance_loss", pocket_distance_loss / sample_size, sample_size, round=3
            )
        coord_loss = sum(log.get("coord_loss", 0) for log in logging_outputs)
        if coord_loss > 0:
            coord_predict = [log.get("coord_predict")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_predict").size(0))]
            coord_target = [log.get("coord_target")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_target").size(0))]
            metrics.log_scalar(
                "coord_loss", coord_loss / sample_size, sample_size, round=3
            )
            rmsd_list = [RMSD(_predict, _target) for _predict,_target in zip(coord_predict, coord_target)]
            metrics.log_scalar(
                "RMSD", np.mean(rmsd_list), sample_size, round=3
            )
        pocket_coord_loss = sum(log.get("pocket_coord_loss", 0) for log in logging_outputs)
        if pocket_coord_loss > 0:
            pocket_coord_predict = [log.get("pocket_coord_predict")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("pocket_coord_predict").size(0))]
            pocket_coord_target = [log.get("pocket_coord_target")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("pocket_coord_target").size(0))]
            metrics.log_scalar(
                "pocket_coord_loss", pocket_coord_loss / sample_size, sample_size, round=3
            )
            rmsd_list = [RMSD(_predict, _target) for _predict,_target in zip(pocket_coord_predict, pocket_coord_target)]
            metrics.log_scalar(
                "RMSD", np.mean(rmsd_list), sample_size, round=3
            )
        prmsd_loss = sum(log.get("prmsd_loss", 0) for log in logging_outputs)
        if prmsd_loss > 0:
            metrics.log_scalar(
                "prmsd_loss", prmsd_loss / sample_size, sample_size, round=3
            )
        pocket_prmsd_loss = sum(log.get("pocket_prmsd_loss", 0) for log in logging_outputs)
        if pocket_prmsd_loss > 0:
            metrics.log_scalar(
                "pocket_rmsd_loss", pocket_prmsd_loss / sample_size, sample_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def RMSD(coord_predict, coord_target):
    mask = coord_target != 0
    rmsd = np.sqrt(np.sum(((coord_predict - coord_target) ** 2) * mask) / (mask[:,0].sum()))
    return rmsd
