# [ICML2024]RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning

This repository is the implementation of RL-CFR in ICML 2024. Please refer to that repo for more documentation.

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:


@inproceedings{
icml2024rlcfr,
title={{RL}-{CFR}: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning},
author={Boning Li, Zhixuan Fang and Longbo Huang},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=pA2Q5Wfspp}
}


## Requirements

libtorch

cmake

## Training

cd src & mkdir build & cd build

cmake -DCMAKE_PREFIX_PATH=[LIBTORCH_DIR] ..

cd .. & cd ..

python train/actor_train.py

## Evaluation

ABSTRACTION_TYPE = 0 --> DEFAULT ACTION ABSTRACTION

ABSTRACTION_TYPE = 1 --> RL-CFR ACTION ABSTRACTION

### Abstraction Evaluation

./src/build/CFR -1 [EVALUATION_NUMBER]

### Exploxity Evaluation

./src/build/CFR -2 [ABSTRACTION_TYPE] [ABSTRACTION_TYPE] [EVALUATION_NUMBER]

### Heads-up Evaluation

./src/build/CFR -3 [THREAD_ID] [ABSTRACTION_TYPE] [ABSTRACTION_TYPE] [EVALUATION_NUMBER]

OR

python train/AI_test.py
