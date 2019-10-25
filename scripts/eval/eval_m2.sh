#!/bin/bash


# rgb
python eval/eval_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 12 \
    --path <path-to-your-model> \
    --vis-interval 1 \
    --cache eval.m2/rgb \
    --num-z-samples 5 \
    --num-iters 100 \

python eval/eval_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 12 \
    --path <path-to-your-model> \
    --vis-interval 1 \
    --cache eval.m2/rgb \
    --num-z-samples 5 \
    --num-iters 100 \


# grayscale
python eval/eval_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 12 \
    --path <path-to-your-model> \
    --vis-interval 1 \
    --cache eval.m2/gray \
    --grayscale \
    --num-z-samples 5 \
    --num-iters 100 \

python eval/eval_multimodal_m2.py \
    --dataset haptix-shepard_metzler_5_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 12 --eval-batch-size 12 \
    --path <path-to-your-model> \
    --vis-interval 1 \
    --cache eval.m2/gray \
    --grayscale \
    --num-z-samples 5 \
    --num-iters 100 \
