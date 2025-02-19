#!/usr/bin/env bash

set -x
GPUS="0"

cd ../

# baseline1 AE-Spherical and KITTI360 dataset

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_baseline/baseline_model_full_sequences_train_val/config.yaml \
#                                 --ckpts /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_baseline/baseline_model_full_sequences_train_val/ckpt-best.pth \
#                                 --num_workers 8 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --test \
#                                 --exp_name baseline_model_full_sequences

# KITTI dataset

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_baseline_kitti/baseline_model_full_sequences_kitti_train_val/config.yaml \
#                                 --ckpts /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_baseline_kitti/baseline_model_full_sequences_kitti_train_val/ckpt-best.pth \
#                                 --num_workers 16 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --test \
#                                 --exp_name baseline_model_full_sequences_kitti

# Ours-SaliencyI2PLoc and KITTI360 dataset

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_contrast/full_model_patch_embed_any_res_1_4_full_query_sequences_train_val/config.yaml \
#                                 --ckpts /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_contrast/full_model_patch_embed_any_res_1_4_full_query_sequences_train_val/ckpt-best.pth \
#                                 --num_workers 16 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --test \
#                                 --exp_name full_model_patch_embed_any_res_1_4_full_query_sequences

# KITTI dataset

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_contrast_kitti/full_model_patch_embed_any_res_full_sequences_kitti_train_val/config.yaml \
#                                 --ckpts /workspace/WorkSpacePR/SaliencyI2PLoc/experiments/CrossModalityRetrieval_contrast_kitti/full_model_patch_embed_any_res_full_sequences_kitti_train_val/ckpt-best.pth \
#                                 --num_workers 16 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --test \
#                                 --exp_name full_model_patch_embed_any_res_full_sequences_kitti