#!/bin/bash


# classification with rgb
python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/rgb/46_parts \
    --path <path-to-your-model> \

python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/rgb/46_parts \
    --path <path-to-your-model> \

python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/rgb/46_parts \
    --path <path-to-your-model> \


# classification with grayscale
python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/gray/46_parts \
    --path <path-to-your-model> \
    --grayscale \

python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model poe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/gray/46_parts \
    --path <path-to-your-model> \
    --grayscale \

python eval/clsinf_multimodal_m2.py \
    --dataset haptix-shepard_metzler_46_parts \
    --model conv-apoe-multimodal-cgqn-v4 \
    --train-batch-size 10 --eval-batch-size 10 \
    --vis-interval 1 \
    --num-z-samples 50 \
    --mod-step 1 \
    --mask-step 1 \
    --cache clsinf.m2.s50/gray/46_parts \
    --path <path-to-your-model> \
    --grayscale \
