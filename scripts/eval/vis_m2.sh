#!/bin/bash

## rgb
python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m2/rgb \
    --num-samples 5 \

python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m2/rgb \
    --num-samples 5 \

python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --cache vis.m2/rgb \
    --num-samples 5 \


## grayscale
python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --grayscale \
    --num-samples 1 \
    --cache vis.m2/gray \

python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --grayscale \
    --num-samples 1 \
    --cache vis.m2/gray \

python eval/vis_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 1 \
    --path <path-to-your-model> \
    --grayscale \
    --num-samples 1 \
    --cache vis.m2/gray \
