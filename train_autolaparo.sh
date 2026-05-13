#!/usr/bin/env bash
# AutoLaparo — TeCNO split: train 01..10, val 11..14, test 15..21.
# Chunk sizes: head=32, fast=32, slow=32.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train.py \
    --dataset autolaparo \
    --data_root ./autolaparo_preprocessed \
    --phase_annotation_dir ./autolaparo_preprocessed/phase_annotations \
    --epochs 100 \
    --tbptt_k 12 \
    --lr_min 1e-6 \
    --head_chunk_size 32 \
    --chunk_size_block 32 \
    --chunk_size_fast_block 32 \
    --chunk_size_slow_block 32 \
    --save_dir ./checkpoints
