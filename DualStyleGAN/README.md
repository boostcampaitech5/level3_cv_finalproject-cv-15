# DualStyleGAN - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In CVPR 2022.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/dualstylegan/) | [**Paper**](https://arxiv.org/abs/2203.13248) | [**Supplementary Video**](https://www.youtube.com/watch?v=scZTu77jixI)
> **Abstract:** *Recent studies on StyleGAN show high performance on artistic portrait generation by transfer learning with limited data. In this paper, we explore more challenging exemplar-based high-resolution portrait style transfer by introducing a novel <b>DualStyleGAN</b> with flexible control of dual styles of the original face domain and the extended artistic portrait domain. Different from StyleGAN, DualStyleGAN provides a natural way of style transfer by characterizing the content and style of a portrait with an <b>intrinsic style path</b> and a new <b>extrinsic style path</b>, respectively. The delicately designed extrinsic style path enables our model to modulate both the color and complex structural styles hierarchically to precisely pastiche the style example. Furthermore, a novel progressive fine-tuning scheme is introduced to smoothly transform the generative space of the model to the target domain, even with the above modifications on the network architecture. Experiments demonstrate the superiority of DualStyleGAN over state-of-the-art methods in high-quality portrait style transfer and flexible style control.*

**Features**:<br> 
**High-Resolution** (1024) | **Training Data-Efficient** (~200 Images) | **Exemplar-Based Color and Structure Transfer**

## Installation

**Clone this repo:**
```bash
git clone https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15.git
cd DualStyleGAN
```
**Dependencies:**

All dependencies for defining the environment are provided in `environment/dualstylegan_env.yaml`.
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda env create -f ./environment/dualstylegan_env.yaml
```
We use CUDA 10.1 so it will install PyTorch 1.7.1 (corresponding to [Line 22](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L22), [Line 25](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L25), [Line 26](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L26) of `dualstylegan_env.yaml`). Please install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/).

## Inference for Style Transfer

Download cartoon checkpoints in https://drive.google.com/drive/folders/1xPo8PcbMXzcUyvwe5liJrfbA5yx4OF1j?usp=sharing

The saved checkpoints are under the following folder structure:
```
checkpoint
|--encoder.pt                     % Pixel2style2pixel model
|--cartoon
    |--generator.pt               % DualStyleGAN model
    |--sampler.pt                 % The extrinsic style code sampling model
    |--exstyle_code.npy           % extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy   % refined extrinsic style codes of Cartoon dataset
...
```

Style transfer cat or dog to princesses in animation
Run style_transfer.py:

```python
python style_transfer.py python style_transfer.py --content ./data/content/{your_picture}
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang2022Pastiche,
  title={Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer},
  author={Yang, Shuai and Jiang, Liming and Liu, Ziwei and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}
```

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).