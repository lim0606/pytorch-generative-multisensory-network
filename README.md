# Generative Multisensory Network
Pytorch implementation of Generative Multisensory Network (GMN) on our paper: 
> Jae Hyun Lim, Pedro O. Pinheiro, Negar Rostamzadeh, Christopher Pal, Sungjin Ahn, [Neural Multisensory Scene Inference](https://arxiv.org/abs/1910.02344) (2019)

<!-- ## Introduction
Please check out [our project website](https://sites.google.com/view/generative-multisensory-net)!
-->

## Getting Started

### Requirements
`python>=3.6`  
`pytorch==0.4.x`  
`tensorflow` (for tensorboardX)  
`tensorboardX`  

### Dataset
data from [MESE](https://github.com/lim0606/multisensory-embodied-3D-scene-environment)  

### Structure
- `data`: data folder
- `datasets`: dataloader definitions
- `models`: model definitions
- `utils`: miscelleneous functions
- `cache`: temporary files
- `eval`: a set of python codes for evaluation / visualization
- `scripts`: scripts for experiments
  ```sh
  ├── eval: eval/visualization scripts are here
  ├── train: training codes are here
  └── train_missing_modalities: training with missing modalities are here
      ├── m5
      ├── m8
      └── m14
  ```
- `main_multimodal.py`: main function to train model

## Experiments
### Train
- For example, you can train an APoE model for vision and haptic data (# of modalities = 2) as follows,  
  ```sh
  python main_multimodal.py \
      --dataset haptix-shepard_metzler_5_parts \
      --model conv-apoe-multimodal-cgqn-v4 \
      --train-batch-size 12 --eval-batch-size 4 \
      --lr 0.0001 \
      --clip 0.25 \
      --add-opposite \
      --epochs 10 \
      --log-interval 100 \
      --exp-num 1 \
      --cache experiments/haptix-m2
  ```  
  For more information, please find example scripts in `scripts/train` folder.  

### Classification (using learned model)
- An example script to run classification with a learned model on held-out date can be written as follows:  
  For the additional Shepard-Metzler objects with 4 or 6 parts (<a href="https://www.codecogs.com/eqnedit.php?latex=|\mathcal{M}|=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|\mathcal{M}|=2" title="|\mathcal{M}|=2" /></a>), 10-way classification.  
  ```sh
  python eval/clsinf_multimodal_m2.py \
  --dataset haptix-shepard_metzler_46_parts \
  --model conv-apoe-multimodal-cgqn-v4 \
  --train-batch-size 10 --eval-batch-size 10 \
  --vis-interval 1 \
  --num-z-samples 50 \
  --mod-step 1 \
  --mask-step 1 \
  --cache clsinf.m2.s50/rgb/46_parts \
  --path <path-to-your-model>
  ```  
  For more information, please find example scripts in `scripts/eval` folder.  

### Train with missing modalities
- If you would like to run an APoE model for <a href="https://www.codecogs.com/eqnedit.php?latex=|\mathcal{M}^{\tt{train}}_S|=\{11,\dots,14\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|\mathcal{M}^{\tt{train}}_S|=\{11,\dots,14\}" title="|\mathcal{M}^{\tt{train}}_S|=\{11,\dots,14\}" /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=|\mathcal{M}|=14" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|\mathcal{M}|=14" title="|\mathcal{M}|=14" /></a>, run following script,  
  ```sh
  python main_multimodal.py \
      --dataset haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-intrapol1114 \
      --model conv-apoe-multimodal-cgqn-v4 \
      --train-batch-size 9 --eval-batch-size 4 \
      --lr 0.0001 \
      --clip 0.25 \
      --add-opposite \
      --epochs 10 \
      --log-interval 100 \
      --cache experiments/haptix-m14-intrapol1114
  ```
  For more information, please find example scripts in `scripts/train_missing_modalities` folder.  

## Contact
For questions and comments, feel free to contact [Jae Hyun Lim](mailto:jae.hyun.lim@umontreal.ca).

## License
MIT License

## Reference
```
@article{jaehyun2019gmn,
  title     = {Neural Multisensory Scene Inference},
  author    = {Jae Hyun Lim and
               Pedro O. Pinheiro and
               Negar Rostamzadeh and
               Christopher J. Pal and
               Sungjin Ahn},
  journal   = {arXiv preprint arXiv:1910.02344},
  year      = {2019},
}
```
