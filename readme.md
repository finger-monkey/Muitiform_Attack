# Cross-Modality Attack Boosted by Gradient-Evolutionary Multiform Optimization (XXXX 2024)

Code for XXXX 2024 paper ``Cross-Modality Attack Boosted by Gradient-Evolutionary Multiform Optimization (XXXX 2024)".


## [Paper](paper/Multiform_attack.pdf)

## [Supplemental Material](paper/Supplementary_Materials.pdf)

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
* To adapt different dataset formats to this code, we have provided conversion scripts. Please refer to CnMix_process.py, cross-modal_dataset_to_market_format.py, deal_SYSU_testset_ID.py, and testset_to_query.py. 


* There is a processed tar file in [BaiduYun](https://pan.baidu.com/s/1dAMc0HEk_xEBQIJD1JWkPA?pwd=kwwu) (Password: kwwu)  with all needed files.

## Preparing Models

* Download re-ID models from [BaiduYun](https://pan.baidu.com/s/1lGoahWk--y-A008zl01VMQ?pwd=k4np) (Password: k4np)


## Run our code
 
See run.sh for more information.

If you find this code useful in your research, please consider citing:

```
@inproceedings{XXXXX,
  title={Cross-Modality Attack Boosted by Gradient-Evolutionary Multiform Optimization},
  author={XXXXXXXXXx},
  booktitle={XXX},
  volume={35},
  number={4},
  pages={3128--3135},
  year={2024}
}
```

## Contact Me

Email: fmonkey625@gmail.com

