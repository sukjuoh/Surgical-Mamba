#!/usr/bin/env bash
# Cholec80 — cuhk4040 split: train 1..40, val/test 41..80.
# Chunk sizes: head=32, fast=64, slow=64.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train.py \
    --dataset cholec80 \
    --data_root ./cholec80_preprocessed \
    --phase_annotation_dir phase_annotations_preprocessed \
    --tool_annotation_dir tool_annotations_preprocessed \
    --head_chunk_size 32 \
    --chunk_size_block 64 \
    --chunk_size_fast_block 64 \
    --chunk_size_slow_block 64 \
    --save_dir ./checkpoints
