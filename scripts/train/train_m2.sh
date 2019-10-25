#!/bin/bash

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix.m2

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix.m2

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix.m2
