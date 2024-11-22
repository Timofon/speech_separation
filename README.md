# Research on speech separation models with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains implementation of models described in research paper.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/project_avss).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

If you want to fine-tune model with augmentations (like our best model), then you need to add `from_pretrained` parameter to trainer section in train config and specify in it the path to your checkpoint. Also you need to change config in `transforms` section to `example_only_instance_augs`.

To run inference (save predictions):

```bash
python inference.py inferencer.save_path='<enter your path>'
```

You need to specify `save_path` according to your file system. Also you need to specify paths to your dataset in inference dataset config.

## How To Measure Metrics

To measure metrics, run the following command based on your directories:

```bash
python metric_eval.py --mix_dir <your dir with mix audio> --s1_pred_dir <your dir with s1 predictions> --s2_pred_dir <your dir with s2 predictions> --s1_gt_dir <your dir with real s1> --s2_gt_dir <your dir with real s2>
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
