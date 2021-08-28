# weakCOPA
The implementation of the paper [Doing Good or Doing Right? Exploring the Weakness of Commonsense Causal Reasoning Models](https://aclanthology.org/2021.acl-short.20.pdf) (ACL2021).

## Install requirements

`conda create -n weakCOPA python=3.7`

`conda activate weakCOPA`

Download the [torch==1.6.0+cu92](https://download.pytorch.org/whl/cu92/torch-1.6.0%2Bcu92-cp37-cp37m-linux_x86_64.whl)

`pip install torch-1.6.0+cu92-cp37-cp37m-linux_x86_64.whl`

`pip install -r requirements.txt`

## Preprocessing

`python preprocess.py`

- get the train/dev/test spilts and test-easy, test-hard set;

- get the challenging set in **EXP1: Perturbation with Distractors** 

  `data/test_adv_rprandom_copa.csv`

  `data/test_adv_rpwrong_copa.csv`

- get the challenging set in **EXP2:  Masking Question Type** 

  `data/test_adv_blind_all_copa.csv`

## BCOPA-CE Test Set

`data/test_ce&ce_reverse.csv` (csv)

or 

`data/BCOPA-CE.xml` (xml)

## Train 

python train.py [model_shortcut] [seed] [adv type] [aug_data]

eg. 

​	finetuning DeBERTa with regularized loss: `python train.py db-l 436 adv 0`

​	finetuning DeBERTa  with augmented BCOPA set: `python train.py db-l 436 noadv bcopa`

