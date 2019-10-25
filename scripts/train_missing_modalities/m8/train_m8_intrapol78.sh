#!/bin/bash

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol78 \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8-intrapol78

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol78 \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8-intrapol78

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol78 \
    --model multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m8-intrapol78
