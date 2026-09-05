# LOCO-AdaMP

## Simulation

**Parameters**

dimension: choices=["low", "high"]

inference: choices=["predci", "coverage", "power"]

data generating model: choices=["independent", "correlated", "nonlinear"]

ml model: choices=["ridge", "randomforest", "kernelridge"]

loss function: choices=["sq", "abs"]

**Example**

python main.py --dim low --inf coverage --simu independent --ml ridge --loss sq
