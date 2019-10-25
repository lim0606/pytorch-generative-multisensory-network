#!/bin/bash

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8
