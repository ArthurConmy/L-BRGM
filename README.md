&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

![diagram](https://i.imgur.com/owOS2K3.jpg)

# StyleGAN-induced data-driven regularization for inverse problems <!-- # STYLEGAN-INDUCED DATA-DRIVEN REGULARIZATION FOR in PROBLEMS -->
### Arthur Conmy, Subhadip Mukherjee, and Carola-Bibiane Schönlieb

This repository is an implementation of the *L-BRGM* model introduced in [our paper](https://arxiv.org/abs/2110.03814). It builds off the implementation of BRGM [here](https://github.com/razvanmarinescu/brgm-pytorch), which in turn is built on top of the StyleGAN2-ADA implementation in PyTorch [here](https://github.com/NVlabs/stylegan2-ada-pytorch).

Currently, the code supports super-resolution and inpainting of faces formatted to the FFHQ dataset standard.

Feel free to get in contact with the primary author with any issues.

## Requirements

Our method, L-BRGM, like the BRGM method, builds on the StyleGAN-ADA Pytorch codebase, so our requirements are the same as for [StyleGAN2 Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch):
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090. If version 11 is not available, the implementation should still work.
* The folder `images1024x1024` from the FFHQ dataset (currently hosted on [drive](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP). Note that shortcuts the folder can be used if working in colab to save lots of downloading).
* Download the pre-trained model `ffhq.pkl` (available [here](https://dl.dropboxusercontent.com/s/jlgybz6nfhmpv54/ffhq.pkl)) and save in the main directory.

## Usage

For further options, call `python3 run.py --help`.

### Super-resolution

To run a superresolution experiment, run

```
python3 run.py --device=cuda --outpath=my_outpath --fpaths=faces/superres/truelow0.png --fpath-corrupted=True --reconstruction-type=superres --input-dim=64 --model=LBRGM
```

### Inpainting 

```
python3 run.py --device=cuda --outpath=my_outpath --fpaths=faces/inpaint/ffhq-1659.png --fpath-corrupted=False --reconstruction-type=inpaint --model=LBRGM --mask=masks/1024x1024/0.png
```

<!-- * Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default. -->
<!-- * 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA DGX-1 with 8 Tesla V100 GPUs. -->
<!-- * For running the inference from a pre-trained model, you need 1 GPU with at least 12GB of memory. We ran on NVIDIA Titan Xp. For training a new StyleGAN2-ADA generator, you need 1-8 GPUS. -->

<!-- Current limitations -->
<!-- No setting seed for reproducible results -->
<!-- No support for DeepGIN or GFPGAN -->`