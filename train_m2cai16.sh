#!/usr/bin/env bash
# M2CAI16-workflow — official split: train workflow_video_01..27, test test_workflow_video_01..14.
# Chunk sizes: head=32, fast=32, slow=32.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train.py \
    --dataset m2cai16 \
    --data_root ./m2cai16_preprocessed \
    --phase_annotation_dir ./m2cai16_preprocessed/phase_annotations \
    --head_chunk_size 32 \
    --chunk_size_block 32 \
    --chunk_size_fast_block 32 \
    --chunk_size_slow_block 32 \
    --save_dir ./checkpoints
