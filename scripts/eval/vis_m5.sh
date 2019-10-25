#!/bin/bash


python eval/vis_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-ul-lr \
    --model multimodal-cgqn-v4 \
    --train-batch-size 1 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m5 \
    --n-context 5 \
    --n-context 10 \

python eval/vis_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-ul-lr \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 1 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m5 \
    --n-context 5 \
    --n-context 10 \
