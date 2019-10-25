#!/bin/bash

python eval/vis_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half \
    --model multimodal-cgqn-v4 \
    --train-batch-size 1 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m14 \
    --n-context 5 \
    --n-context 10 \
    --n-mods 1 \
    --n-mods 2 \
    --n-mods 3 \
    --n-mods 4 \

python eval/vis_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 1 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m14 \
    --n-context 5 \
    --n-context 10 \
    --n-mods 1 \
    --n-mods 2 \
    --n-mods 3 \
    --n-mods 4 \
