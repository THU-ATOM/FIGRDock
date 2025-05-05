# FIGRDock: Fast Interaction-Guided Regression for Flexible Docking

This is the official code implementation for the paper: "FIGRDock: Fast Interaction-Guided Regression for Flexible Docking".

## Pretraining Models and Data Download

**Link:** [Insert Download Link Here]

**File Explanation:**

* `cond_checkpoint11.pt`: Conditional Encoder checkpoint.
* `figrdock/checkpoint_last.pt`: Final encoder checkpoint.
* `unimol_bindnet_8A_all_with_Atom_apo`: Training data.
* `unimol_bindnet_8A_all_with_Atom_test_good_data`: Testing data.

## Training on PDBBind



```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 10099 unicore_train.py --user-dir ./unimol /data/protein/SKData/unimol_bindnet_8A_all_with_Atom_apo --train-subset train --valid-subset valid --num-workers 8 --ddp-backend c10d --task docking_pose_v2 --loss flexible_docking_pose_v2 --arch docking_pose_v2 --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 --lr-scheduler polynomial_decay --lr 3e-4 --warmup-ratio 0.06 --max-epoch 100 --batch-size 4 --mol-pooler-dropout 0.2 --pocket-pooler-dropout 0.2 --update-freq 1 --seed 42 --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir ./figrdock/tsb --log-interval 1 --log-format simple --validate-interval 10 --keep-last-epochs 10 --best-checkpoint-metric valid_loss --patience 2000 --all-gather-list-size 1024000 --finetune-model /data/protein/SKData/UniMOLData/checkpoint11.pt --dist-threshold 8.0 --recycling 4 --save-dir ./figrdock --find-unused-parameters --required-batch-size-multiple 1 --pocket-dict dict_sidechain.txt --use-flexible-docking flex_all  --max-pocket-atoms 510 --limit_dataset_val_size 10 --freeze_encoder 0 --pocket_loss_weight 1 --sc_aug 0 --dock_interpolation 0 --coord_decode_total_iter 6  --geom_reg_steps 0 --coord_decode_layers 8 --las_init_opti 0 > figrdock.log 2>&1 &
```


## Testing On PDBBind

### HOLO protein input

```bash
CUDA_VISIBLE_DEVICES=6 python -u pdbbind_benchmark.py --data_path /data/protein/SKData/DiffDock-Pocket/data/PDBBIND_atomCorrected/ --test-lmdb /data/protein/SKData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_holo.lmdb --device 6 --checkpoint /data/protein/SKData/UniMOL_Docking/UniMol_Docking/unimol_docking_v2/figrdock/checkpoint_last.pt --use-flexible-docking rigid  --task felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_las_iter6_layers8_bz4_woprmsd_holo_good_data_noclashfix_iters500_last_fb --max-pocket-atoms 510  --coord_decode_total_iter 6 --coord_decode_layers 8 --no-clash-fix --geom_reg_steps 500  > felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_las_iter6_layers8_bz4_woprmsd_holo_good_data_noclashfix_iters500_last_fb.log 2>&1 &
```

### APO protein input

```bash
CUDA_VISIBLE_DEVICES=7 python -u pdbbind_benchmark.py --test-lmdb /data/protein/SKData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_apo.lmdb --device 7 --checkpoint /data/protein/SKData/UniMOL_Docking/UniMol_Docking/unimol_docking_v2/figrdock/checkpoint_last.pt --use-flexible-docking flex_all --sc_aug 0 --pocket-dict dict_sidechain.txt  --task felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_las_iter6_layers8_bz4_woprmsd_apo_good_data_noclashfix_iters500_last_fix_bigring  --max-pocket-atoms 510 --coord_decode_total_iter 6  --coord_decode_layers 8  --no-clash-fix --geom_reg_steps 500  > felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_las_iter6_layers8_bz4_woprmsd_apo_good_data_noclashfix_iters500_last_fix_bigring.log 2>&1 &
```





