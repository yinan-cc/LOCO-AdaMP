# LOCO-AdaMP

## Simulation

**Parameters**

dimension: choices=["low", "high"]

inference: choices=["predci", "coverage", "power"]

data generating model: choices=["independent", "correlated", "nonlinear"]

ml model: choices=["ridge", "randomforest", "kernelridge"]

**Example**

python main.py --dim low --inf predci --simu independent --ml ridge

## Case Study

**Run the following files in order**

case_study_loco_vim.py

case_study_cpi.R

case_study_tmse.py
