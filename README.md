# Lossy Image Compression with Quantized Hierarchical VAEs
QRes-VAE is a lossy image coder.

## Install
**Requirements**: the code has been tested on Windows and Linux with the following packages:
- `pytorch=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI)), `timm=0.5.4` ([link](https://github.com/rwightman/pytorch-image-models)).

**Download**:
1. Download the repository;
2. Download the pre-trained model checkpoints and put them in the `qres-vae/checkpoints` folder.

### Pre-trained model checkpoints
- QRes-VAE (34M) [[Google Drive](https://drive.google.com/file/d/1qBJ306VgSbwo7eWWxqYnQI0bRhY0l-7R/view?usp=sharing)]: our main model for natural image compression.
- QRes-VAE (17M) [Google Drive]: a smaller model trained on CelebA64 dataset. For ablation study and additional experiments only.

## Usage
### Image compression
- **Compression and decompression**: See `demo.ipynb`.
- TBD
### As a generative model
- TBD
- TBD

## Evaluation
- Rate-distortion curve on images:
- BD-rate:

## Training
TBD

## License
TBD

## Citation
TBD
