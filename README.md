# Mining False Positive Examples for Text-Based Person Re-indentification

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/taksau/GPS-Net/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5.0-%237732a8)

We provide the code for reproducing result of our  paper **Mining False Positive Examples for Text-Based Person Re-indentification**.

## Getting Started
#### Dataset Preparation

**CUHK-PEDES**

   Organize them in `dataset` folder as follows:
       

   ~~~
   |-- dataset/
   |   |-- <CUHK-PEDES>/
   |       |-- <CUHK-PEDES>/ 
   |           |-- imgs
                   |-- cam_a
                   |-- cam_b
                   |-- ...
   |           |-- reid_raw.json
   
   ~~~

   Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) and then run the `process_CUHK_data.py` as follow:

   ~~~
   cd MFPE
   python ./dataset/process_CUHK_data.py
   ~~~
#### Training and Testing
   ~~~
   cd MFPE/src
   python train.py
   ~~~
#### Evaluation
   We provide the [best models](https://drive.google.com/file/d/1WK1aCDYtNnfJySRoJAqX6mV2ODm_J1a0/view?usp=share_link) for evaluation.
   ~~~
   cd MFPE/src
   python test.py --resume=best_64.59.pth
   ~~~
