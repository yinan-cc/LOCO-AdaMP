# LOCO-AdaMP

## Parameters
dimension: choices=["low", "high"]

inference: choices=["predci", "coverage", "power"]

data generating model: choices=["independent", "correlated", "nonlinear"]

ml model: choices=["ridge", "randomforest", "kernelridge"]

## Example

python main.py --dim low --inf predci --simu independent --ml ridge
