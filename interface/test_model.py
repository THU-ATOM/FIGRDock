# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/admin01/miniconda3/envs/UniMOL/lib:$LD_LIBRARY_PATH
#  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/admin01/miniconda3/envs/UniMol/lib:$LD_LIBRARY_PATH
import os

import sys

test_dirs = ['/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_1k_train',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_1k_train_freeze_encoder',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_1k_train_freeze_encoder_lr_1e-4',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_1k_only_pklw_0.1',
             
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_only_pklw_0.1',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_only_ligand_loss_noscaug',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_train_freeze_encoder_lr_1e-4',
             ]


test_dirs = [
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_only_pklw_0.1',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_only_ligand_loss_noscaug',
             '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_1k_train_freeze_encoder_lr_1e-4',
             ]

test_dirs = [
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11'
]

test_dirs = [
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_plw0.5'
]
test_dirs = [
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_all_train'
]
# test_dirs = [
#     '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train',
# ]

# test_dirs = [
#     '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_all_train'
# ]

# --pocket-dict dict_pkt.txt 


test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_baseline_all_train',
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_plw0.5',
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11'
}


test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_first_v1'
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_freeze_pencoder',
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_pkw0.1',
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_pkw0.1_apo'
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_freeze_encoder'
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc'
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc_epoch200'
}

test_dirs = {
    '/mnt/nfs-ssd/data/fengshikun/UniMol_Docking_sc/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean'
}

# test_dirs = {
#     '/nfs/SKData/UniMol_Docking_sc/UniMol_Docking/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_interpolation_alpha1_beta3'
# }

test_dirs = {
    '/nfs/SKData/UniMol_Docking_sc/UniMol_Docking/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_interpolation_alpha1_beta10'
}

test_dirs = {
    '/nfs/SKData/UniMol_Docking_sc/UniMol_Docking/unimol_docking_v2/felix_bindnet_pretrain_all_train_ep11_apo_nosc_fix_mean_las_iter6'
}

# test_cmd = 'CUDA_VISIBLE_DEVICES=5 python -u pdbbind_benchmark.py --test-lmdb /mnt/nfs-ssd/data/fengshikun/unimol_bindnet_8A_all_with_Atom_test/test_holo.lmdb --device 5 --checkpoint {}/checkpoint_last.pt --use-flexible-docking rigid  --task {} > {}.log 2>&1 &'



# --pocket-dict dict_pkt.txt
# test_cmd2 = 'CUDA_VISIBLE_DEVICES=4 python -u pdbbind_benchmark.py --test-lmdb /mnt/nfs-ssd/data/fengshikun/unimol_bindnet_8A_all_with_Atom_test/test_holo.lmdb --device 6 --checkpoint {}/checkpoint_best.pt --use-flexible-docking rigid  --task {}_best   > {}_best.log 2>&1 &'

test_cmd = 'CUDA_VISIBLE_DEVICES=2 python -u pdbbind_benchmark.py --test-lmdb /nfs/SKData/AIRDockData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_holo.lmdb --device 6 --checkpoint {}/checkpoint_last.pt --use-flexible-docking rigid  --task {}_max_pock_510_good_data --max-pocket-atoms 510 > {}_max_pock_510_good_data.log 2>&1 &'

test_cmd = 'CUDA_VISIBLE_DEVICES=2 python -u pdbbind_benchmark.py --test-lmdb /nfs/SKData/AIRDockData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_holo.lmdb --device 0 --checkpoint {}/checkpoint_last.pt --use-flexible-docking rigid  --task {}_max_pock_510_good_data --max-pocket-atoms 510 --no-clash-fix --geom_reg_steps 1 > {}_max_pock_510_good_data_noclashfix.log 2>&1 &'


test_cmd_flex = 'CUDA_VISIBLE_DEVICES=5 python -u pdbbind_benchmark.py --test-lmdb /mnt/nfs-ssd/data/fengshikun/unimol_bindnet_8A_all_with_Atom_test/test_apo.lmdb --device 7 --checkpoint {}/checkpoint_best.pt --use-flexible-docking flex_all --sc_aug 0 --pocket-dict {}  --task {}  > {}_best.log 2>&1'

test_cmd_flex = 'CUDA_VISIBLE_DEVICES=1 python -u pdbbind_benchmark.py --test-lmdb /nfs/SKData/AIRDockData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_apo.lmdb --device 7 --checkpoint {}/checkpoint_last.pt --use-flexible-docking flex_all --sc_aug 0 --pocket-dict {}  --task {}_good_data  --max-pocket-atoms 510 > {}_last.log 2>&1 &'


test_cmd_flex = 'CUDA_VISIBLE_DEVICES=1 python -u pdbbind_benchmark.py --test-lmdb /nfs/SKData/AIRDockData/unimol_bindnet_8A_all_with_Atom_test_good_data/test_apo.lmdb --device 3 --checkpoint {}/checkpoint_last.pt --use-flexible-docking flex_all --sc_aug 0 --pocket-dict {}  --task {}_good_data  --max-pocket-atoms 510 --no-clash-fix --geom_reg_steps 1 > {}_last_noclash_fix.log 2>&1 &'

for test_dir in test_dirs:
    task_name = os.path.basename(test_dir)
    task_name += '_apo'
    test_log_file = task_name + '_apo_test'
    if 'pretrain' in test_dir:
        # cmd = test_cmd_flex.format(test_dir, 'dict_pkt.txt', task_name, test_log_file)
        cmd = test_cmd_flex.format(test_dir, 'dict_sidechain.txt', task_name, test_log_file)
    else:
        cmd = test_cmd_flex.format(test_dir, 'dict_pkt.txt', task_name, test_log_file)

    print(cmd)
    os.system(cmd)
    


for test_dir in test_dirs:
    task_name = os.path.basename(test_dir)
    test_log_file = task_name + '_holo_test'
    cmd = test_cmd.format(test_dir, task_name, test_log_file)
    print(cmd)
    os.system(cmd)
#     task_name = f'{task_name}_best'
#     cmd2 = test_cmd2.format(test_dir, task_name, test_log_file)
#     print(cmd2)
#     os.system(cmd2)