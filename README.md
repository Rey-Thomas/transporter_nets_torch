# PyTorch adaptation of [Ravens - Transporter Networks](https://github.com/google-research/ravens)

- ### [Original repository (in TensorFlow)](https://github.com/google-research/ravens)
- ### Original Paper: Transporter Networks: Rearranging the Visual World for Robotic Manipulation
  [Project Website](https://transporternets.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2010.14406.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2020
  *Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong,<br>Ivan Krasin, Dan Duong, Vikas Sindhwani, Johnny Lee*

### Project Description
Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation, with emphasis on pick and place.
It features a Gym-like API with 10 tabletop rearrangement tasks, each with (i) a scripted oracle that provides expert demonstrations (for imitation learning), and (ii) reward functions that provide partial credit (for reinforcement learning).
The goal of this project is to see how the depth information impact the performance of the Transporter architecture. 
We add to the original repository 3 models that can be used:
  - no-depth-transporter
  - adabins-nyu-transporter
  - adabins-kitti-transporter

The first model is the same as the baseline but do not use the depth map.
The two last models are using a monocular depth estimator, based on the Adabins architecture from https://doi.org/10.1109/CVPR46437.2021.00400 . All the code of the article can be seen on https://github.com/shariqfarooq123/AdaBins . The difference between the two model is which pretrained model of adabins is used. NYU is an indoor dataset. Kitii is an outdoor dataset. 

This project will compare 3 tasks.


<img src="https://github.com/Rey-Thomas/transporter_nets_torch/blob/main/task.png" /><br>

(a) **block-insertion**: pick up the L-shaped red block and place it into the L-shaped fixture.<br>
(b) **towers-of-hanoi**: sequentially move disks from one tower to another—only smaller disks can be on top of larger ones.<br>
(c) **stack-block-pyramid**: sequentially stack 6 blocks into a pyramid of 3-2-1 with rainbow colored ordering.<br>



## Installation

**Step 1.** Create a Conda environment with Python 3, then install Python packages:

```shell
make install
```

Or

```shell
cd ~/transporter_nets_torch
conda create --name ravens_torch python=3.7 -y
conda activate ravens_torch
pip install -r requirements.txt
python setup.py install --user
```

**Step 2.** Export environment variables in your terminal

```shell
export RAVENS_ASSETS_DIR=`pwd`/ravens_torch/environments/assets/;
export WORK=`pwd`;
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Getting Started

**Step 1.** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens_torch/demos.py --disp=True --task=block-insertion --mode=train --n=10
python ravens_torch/demos.py --disp=True --task=block-insertion --mode=test --n=100
```

You can also manually change the parameters in `ravens_torch/demos.py` and then run `make demos` in the shell (see the Makefile if needed).

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 2.** Train a model e.g., Transporter Networks model. Model checkpoints are saved to the `data/checkpoints` directory. Optional: you may exit training prematurely after 1000 iterations to skip to the next step.

```shell
python ravens_torch/train.py --task=block-insertion --agent=transporter --n_demos=10
```

Likewise for demos, you can run `make train`.

**Step 3.** Evaluate a Transporter Networks agent using the model trained for 1000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens_torch/test.py --disp=True --task=block-insertion --agent=transporter --n_demos=10 --n_steps=1000
```

Again, `make test` automates it.

**Step 4.** Plot and print results with `make plot` or:

```shell
python ravens_torch/plot.py --disp=True --task=block-insertion --agent=transporter --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

## Datasets

Download generated train and test datasets from the original authors of the paper:

```shell
wget https://storage.googleapis.com/ravens-assets/block-insertion.zip
wget https://storage.googleapis.com/ravens-assets/towers-of-hanoi.zip
wget https://storage.googleapis.com/ravens-assets/stack-block-pyramid.zip
```


