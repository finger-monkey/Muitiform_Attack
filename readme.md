# Multiform Attack for Transferable Cross-Modal Person Re-Identification (NeurIPS 2026)

Code for the NeurIPS 2024 paper **"Multiform Attack for Transferable Cross-Modal Person Re-Identification"**.

## Paper

- [Paper](paper/Multiform_attack.pdf)

## Environment

Please reproduce the software environment using the configuration files provided in this repository. The recommended entry point is [`environment.yml`](environment.yml). Additional dependency and installation information is available in:

- [`requirements.txt`](requirements.txt)
- [`requirements-pip.txt`](requirements-pip.txt)
- [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)
- [`environment-summary.md`](environment-summary.md)
- [`setup_environment.sh`](setup_environment.sh)

The recorded environment uses Python 3.9, PyTorch 1.12, torchvision 0.13, CUDA 11.3, and FAISS GPU 1.7.3. Please follow the supplied configuration as closely as possible. Changes to Python, CUDA, PyTorch, torchvision, FAISS, NumPy, or related library versions may alter numerical behavior, model loading, data processing, and final evaluation results. An environment that differs from the supplied configuration may therefore produce results inconsistent with the reported reproduction.

A typical setup is:

```bash
conda env create -f environment.yml
conda activate pami-attack
```

Alternatively, consult `ENVIRONMENT_SETUP.md` before using the pip requirement files.

## Preparing Data

The experiments use Market-1501 (transformed into CnMix), Sketch-ReID, SYSU-MM01, RegDB, and related cross-modality data. After downloading the datasets, organize them under a common data root and update the example paths in the processing scripts before running them.

A processed dataset archive is available from [BaiduYun](https://pan.baidu.com/s/1dAMc0HEk_xEBQIJD1JWkPA?pwd=kwwu) (Password: `kwwu`).

### Data processing scripts

- [`CnMix_process.py`](CnMix_process.py) constructs CnMix-style images from Market-1501. It probabilistically generates grayscale images and RGB combinations containing grayscale or sketch channels while preserving the source directory organization.
- [`cross-modal_dataset_to_market_format.py`](cross-modal_dataset_to_market_format.py) converts images from identity/camera subdirectories into a flat Market-1501-style naming scheme such as `PID_camera_sequence_01.jpg`.
- [`deal_SYSU_testset_ID.py`](deal_SYSU_testset_ID.py) reads the SYSU test identity list and moves the corresponding identity directories into the designated test split.
- [`testset_to_query.py`](testset_to_query.py) samples part of each identity's test images and moves them into a separate query directory, providing a simple query/gallery organization for evaluation.

These scripts contain example local paths and may move or copy files. Review and replace all input/output paths, keep an untouched copy of the raw datasets, and verify the generated identity and camera labels before training.

Other dataset-specific conversion support is also provided, including [`convert_llcm_to_market.py`](convert_llcm_to_market.py) for converting LLCM into the project-compatible layout.

## Preparing Models

Pretrained ReID models are available from [BaiduYun](https://pan.baidu.com/s/1lGoahWk--y-A008zl01VMQ?pwd=k4np) (Password: `k4np`).

Download the required checkpoints and update the checkpoint paths passed to the attack program.

## Running the Code

See [`run.sh`](run.sh) for example commands. Before execution, replace the placeholder data and checkpoint paths with the corresponding locations on your machine.

```bash
bash run.sh
```

For a reproducible comparison, keep the dataset split, model checkpoint, random seed, perturbation budget, image size, batch size, and environment versions fixed across runs.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{XXXXX,
  title={Multiform Attack for Transferable Cross-Modal Person Re-Identification},
  author={XXXXXXXXXx},
  booktitle={XXX},
  volume={35},
  number={4},
  pages={3128--3135},
  year={2024}
}
```

## Contact

Email: fmonkey625@gmail.com

### &#8627; Visitors

[![Visit tracker](https://clustrmaps.com/map_v2.png?cl=ffffff&w=896&t=tt&d=zLtXBhTnXw66l00fakOMI4K9BJmzjJ_0hpftLgebA_Y)](https://clustrmaps.com/site/1c4pf)
