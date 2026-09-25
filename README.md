# LOCO-AdaMP

This repository contains the code for the paper “LOCO-AdaMP: Built-in LOCO Inference for Adaptive
Minipatch Ensembles with Enhanced Prediction” by Yinan Cheng and Lili Zheng

## Simulation Study

**Parameters**

dimension: choices=["low", "high"]

inference: choices=["predci", "coverage", "power"]

data generating model: choices=["independent", "correlated", "nonlinear"]

ml model: choices=["ridge", "randomforest", "kernelridge"]

loss function: choices=["sq", "abs"]

**Example**

python main.py --dim low --inf coverage --simu independent --ml ridge --loss sq

## Case Study
