# Lossy Image Compression with Quantized Hierarchical VAEs
QRes-VAE (Quantized ResNet VAE) is a neural network model for lossy image compression.
It is based on the ResNet VAE architecture.

Arxiv report: TBD


## Install
**Requirements**:
- Python, `pytorch>=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI)), `timm>=0.5.4` ([link](https://github.com/rwightman/pytorch-image-models)).
- Code has been tested in the following environments:
    - Both Windows and Linux, with Intel CPUs and Nvidia GPUs
    - Python 3.9
    - `pytorch=1.9, 1.10, 1.11` with CUDA 11.3
    - `pytorch=1.12` with CUDA 11.6. Models run faster (both training and testing) in this version than in previous versions.


**Download**:
1. Download the repository;
2. Download the pre-trained model checkpoints and put them in the `checkpoints` folder. See `checkpoints/README.md` for expected folder structure.


## Pre-trained models
- QRes-VAE (34M) [[Google Drive](https://drive.google.com/file/d/1qBJ306VgSbwo7eWWxqYnQI0bRhY0l-7R/view?usp=sharing)]: our main model for natural image compression.
- QRes-VAE (17M) [[Google Drive](https://drive.google.com/file/d/1p8GpOxfb5r0Hoe_eCfUx3JLq8AmtD5AW/view?usp=sharing)]: a smaller model trained on CelebA dataset for ablation study.
- QRes-VAE (34M, lossless)

The `lmb` in the name of folders is the multiplier for MSE during training. I.e., `loss = rate + lmb * mse`.
A larger `lmb` produces a higher bit rate but lower distortion.


## Usage
### Image compression
- **Compression and decompression (lossy)**: See `demo.ipynb`.
- **Compression and decompression (lossless)**:
### As a VAE generative model
- **Progressive decoding**:
- **Sampling**:
- **Latent space interpolation**:
- **Inpainting**:


## Evaluation
- Rate-distortion curve:
- BD-rate:


## Training
Training is done by minimizing the `stats['loss']` return by the model's `forward` function.

### Single GPU training
The file `train.py` is a simple example script for single-GPU training.
To train the `qres34m` model with `lmb=1024`:
```
python train.py --model qres34m --lmb 1024 --train_root /path/to/coco/train2017 --train_crop 256 \
--val_root /path/to/kodak --batch_size 64 --workers 4
```
In case of a `CUDA error: out of memory`, try reduce the batchsize (better with learning rate as well):
```
python train.py --model qres34m --lmb 1024 --train_root /path/to/coco/train2017 --train_crop 256 \
--val_root /path/to/kodak --batch_size 16 --lr 1e-4 --workers 4
```

### Multi-GPU training (TBD)
The `dev/train.py` supports multi-GPU training and can reproduce the paper's training results.
<!-- Check `dev/README.md` for instructions. -->
<!-- This requires the `wandb` package for logging. -->
Train on 4 GPUs:
```
torchrun --nproc_per_node 4 dev/train.py --model qres34m --model_args lmb=128 --trainsets coco-train \
--transform crop=256,hflip=True --valset kodak --val_bs 1 --batch_size 16 --lr 2e-4 --lr_sched constant \
--grad_clip 2 --epochs 400 --fixseed --workers 4 \
--wbproject qres34m --wbgroup qres34m-coco-256 --wbmode online --name lmb128
```


## License
TBD


## Citation
TBD
