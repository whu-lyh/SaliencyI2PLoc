#!/usr/bin/env bash
clear
GPUS="0"

cd ../

# KITTI360

#  CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/config/CrossModalityRetrieval_baseline.yaml \
#                                 --num_workers 12 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --exp_name baseline_model_full_sequences

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                  --config /workspace/WorkSpacePR/SaliencyI2PLoc/config/CrossModalityRetrieval_contrast_perspective.yaml \
#                                  --num_workers 12 \
#                                  --val_freq 1 \
#                                  --gpu 0 \
#                                  --refactor \
#                                  --exp_name full_model_patch_embed_any_res_full_sequences

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/config/CrossModalityRetrieval_contrast.yaml \
#                                 --num_workers 12 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --exp_name full_model_patch_embed_any_res_full_sequences

# kitti dataset

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/config/CrossModalityRetrieval_baseline_kitti.yaml \
#                                 --num_workers 16 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --exp_name baseline_model_full_sequences

# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --config /workspace/WorkSpacePR/SaliencyI2PLoc/config/CrossModalityRetrieval_contrast_kitti.yaml \
#                                 --num_workers 16 \
#                                 --val_freq 1 \
#                                 --gpu 0 \
#                                 --refactor \
#                                 --exp_name full_model_patch_embed_any_res_full_sequences