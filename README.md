# Lossy Image Compression with Quantized Hierarchical VAEs
QRes-VAE (Quantized ResNet VAE) is a neural network model that can do lossy image compression.
It is based on a hierarchical VAE architecture.


## Install
**Requirements**:
- `pytorch>=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI)), `timm>=0.5.4` ([link](https://github.com/rwightman/pytorch-image-models)).
- Code has been tested on Windows and Linux with Intel CPUs and Nvidia GPUs (Python 3.9, CUDA 11.3).


**Download**:
1. Download the repository;
2. Download the pre-trained model checkpoints and put them in the `qres-vae/checkpoints` folder.


## Pre-trained model checkpoints
- QRes-VAE (34M) [[Google Drive](https://drive.google.com/file/d/1qBJ306VgSbwo7eWWxqYnQI0bRhY0l-7R/view?usp=sharing)]: our main model for natural image compression.
- QRes-VAE (17M) [[Google Drive](https://drive.google.com/file/d/1p8GpOxfb5r0Hoe_eCfUx3JLq8AmtD5AW/view?usp=sharing)]: a smaller model trained on CelebA dataset for ablation study.


## Usage
### Basic image compression
- **Compression and decompression (lossy)**: See `demo.ipynb`.
- **Compression and decompression (lossless)**:
### As a VAE generative model
- **Progressive decoding**:
- **Sampling**:
- **Latent space interpolation**:
- **Inpainting**:


## Evaluation
- Rate-distortion curve on images:
- BD-rate:


## Training
The file `train.py` is a clean (but less flexible) single GPU training script that can *approximately* reproduce our results.
### Lossy comprssion
To train the `qres34m` model with `lmb=1024`,
```
python train.py --model qres34m --lmb 1024 --train_root /path/to/coco/train2017 --train_crop 256 \
--val_root /path/to/kodak --batch_size 64 --workers 4
```
However, this will probably give a `CUDA error: out of memory` unless your GPU has 32G RAM. To train with a smaller batchsize:
```
python train.py --model qres34m --lmb 1024 --train_root /path/to/coco/train2017 --train_crop 256 \
--val_root /path/to/kodak --batch_size 32 --lr 1e-4 --workers 4
```
### Lossless comprssion
To train the `qres34m_ll` model:
```
python train.py --model qres34m_ll --train_root /path/to/coco/train2017 --train_crop 256 \
--val_root /path/to/kodak --batch_size 64 --epochs 200 --workers 4
```
Again, please adjust the batch size and learning rate according to your available GPU RAM.


## Exactly reproduce the paper training
TODO: `dev` branch
This requires the `wandb` package for logging.

Train on 4 GPUs:
```
torchrun --nproc_per_node 4 train2.py --model qres34m --model_args lmb=128 --trainsets coco-train \
--transform crop=256,hflip=True --valset kodak --val_bs 1 --batch_size 16 --lr 2e-4 --lr_sched constant \
--grad_clip 2 --epochs 400 --fixseed --workers 4 \
--wbproject qres34m --wbgroup qres34m-coco-256 --wbmode online --name lmb128
```


## License
TBD


## Citation
TBD
