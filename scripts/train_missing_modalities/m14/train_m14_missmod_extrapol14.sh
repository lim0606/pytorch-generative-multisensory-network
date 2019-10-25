#!/bin/bash

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol14 \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 9 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m14-extrapol14

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol14 \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 9 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m14-extrapol14

python main_multimodal.py \
    --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol14 \
    --model multimodal-cgqn-v4 \
    --train-batch-size 9 --eval-batch-size 4 \
    --lr 0.0001 \
    --clip 0.25 \
    --add-opposite \
    --epochs 10 \
    --log-interval 100 \
    --cache experiments/haptix-m14-extrapol14
