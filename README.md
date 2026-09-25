# LOCO-AdaMP

This repository contains the code for the paper “LOCO-AdaMP: Built-in LOCO Inference for Adaptive
Minipatch Ensembles with Enhanced Prediction” by Yinan Cheng and Lili Zheng

## Simulation Study

**Parameters**

Dimension: choices=["low", "high"]
"low": the low dimensional setting (M =50, N =200)
"high": the high dimensional setting (M =500, N =200)

inference: choices=["predci", "coverage", "power"]
"predci": obtain figures for test errors, LOCO inference confidence intervals, LOCO-AdaMP targets and Assumption 5 checking
"coverage": obtain figures for LOCO-AdaMP coverage rates
"power": obtain figures for LOCO powers

data generating model: choices=["independent", "correlated", "nonlinear"]
"independent": linear independent model
"correlated": linear correlated model
"nonlinear": nonlinear independent model

ml model: choices=["ridge", "randomforest", "kernelridge"]
"randomforest": decision tree for LOCO-MP and LOCO-AdaMP; random forest for LOCO-Split

loss function: choices=["sq", "abs"]
"sq": squared loss
"abs": absolute loss

**Example**

python main.py --dim low --inf coverage --simu independent --ml ridge --loss sq

## Case Study
