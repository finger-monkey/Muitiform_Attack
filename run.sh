
CUDA_VISIBLE_DEVICES=1 python -W ignore Multiform_attack.py -s sysu_v2 -m CnMix -m2 CnMix -t sysu_v2 --data /sda1/gyp/data --batch_size 128\
 --resume /sda1/gyp/DDAG/save_model/sysu_G_P_3_drop_0.2_4_8_lr_0.1_seed_0_best.t\
 --resumeSearchTgt /sda1/gyp/DDAG/save_model/CnMix.t\
 --resumeSearchTgt2 /sda1/gyp/DDAG/save_model/CnMix.t\
 --resumeTgt /sda1/gyp/DDAG/save_model/regdb_G_P_3_drop_0.2_4_8_lr_0.1_seed_0_trial_1_best.t

 CUDA_VISIBLE_DEVICES=1 python -W ignore Multiform_attack.py -s regdb_v2 -m CnMix -m2 CnMix -t regdb_v2 --data /sda1/gyp/data --batch_size 64\
 --resume /sda1/gyp/DDAG/save_model/regdb_G_P_3_drop_0.2_4_8_lr_0.1_seed_0_trial_1_best.t\
 --resumeSearchTgt /sda1/gyp/DDAG/save_model/CnMix.t\
 --resumeSearchTgt2 /sda1/gyp/DDAG/save_model/CnMix.t\
 --resumeTgt /sda1/gyp/DDAG/save_model/regdb_G_P_3_drop_0.2_4_8_lr_0.1_seed_0_trial_1_best.t

