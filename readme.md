# XXXXXXXXXXXXXXXXXXXXX (XXXX 2024)

Code for XXXX 2024 paper ``XXXXXXXXXXXXXXXXXX (XXXX 2024)".


## [Paper](pdfs/XXXX.pdf)

## [Supplemental Material](pdfs/MetaAttack_Supp.pdf)

## Requirements:
* python 3.7
* CUDA==10.1
* Market1501 (will transform to CnMix), Sketch-ReID, SYSU and RegDB dataset.
* faiss-gpu==1.6.0
* Other necessary packages listed in [requirements.txt](requirements.txt)

## Preparing Data

* Clone our repo

Market-1501(namely CnMix) (SYSU and RegDB are the same):
* Download "Market-1501-v15.09.15.zip".
* Create a new directory, rename it as "data".
* Create a directory called "raw" under "data" and put "Market-1501-v15.09.15.zip" under it.
* The processed dataset is provided in the link below, please refer to it.


* There is a processed tar file in [BaiduYun](https://pan.baidu.com/s/1dAMc0HEk_xEBQIJD1JWkPA?pwd=kwwu) (Password: kwwu)  with all needed files,
  password: xxxx. You can directly put it under "data" and unzip it.

## Preparing Models

* Download re-ID models from [BaiduYun](https://pan.baidu.com/s/1lGoahWk--y-A008zl01VMQ?pwd=k4np) (Password: k4np)


* Put models under logs/{datasetname}

## Run our code
 
See attack.sh for more information.

If you find this code useful in your research, please consider citing:

```
@inproceedings{XXXXX,
  title={XXXXg},
  author={Yang, Fengxiang and Zhong, Zhun and Liu, Hong and Wang, Zheng and Luo, Zhiming and Li, Shaozi and Sebe, Nicu and Satoh, Shin’ichi},
  booktitle={XXX},
  volume={35},
  number={4},
  pages={3128--3135},
  year={2024}
}
```

## Acknowledgments

Our code is based on [UAP-Retrieval](https://github.com/theFool32/UAP_retrieval), 
if you use our code, please also cite their paper.
```
@inproceedings{Li_2019_ICCV,
    author = {Li, Jie and Ji, Rongrong and Liu, Hong and Hong, Xiaopeng and Gao, Yue and Tian, Qi},
    title = {Universal Perturbation Attack Against Image Retrieval},
    booktitle = {ICCV},
    year = {2019}
}
```
## Contact Me

Email: fmonkey625@gmail.com

