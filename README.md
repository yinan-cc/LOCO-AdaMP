# LOCO-AdaMP

This repository contains the code for [LOCO-AdaMP: Built-in LOCO Inference for Adaptive
Minipatch Ensembles with Enhanced Prediction](link) by Yinan Cheng and Lili Zheng

## Simulation Study

### Parameters

| Argument | Choices | Description |
| --- | --- | --- |
| `--dim` | `low`, `high` | Low dimensional: \(M=50, N=200\); high dimensional: \(M=500, N=200\). |
| `--inf` | `predci`, `coverage`, `power` | `predci`: test errors, LOCO confidence intervals, LOCO-AdaMP targets, and Assumption 5 checks; `coverage`: LOCO-AdaMP coverage rates; `power`: power of LOCO tests. |
| `--simu` | `independent`, `correlated`, `nonlinear` | Linear model with independent features, linear model with one correlated pair of features, or nonlinear model with independent features. |
| `--ml` | `ridge`, `randomforest`, `kernelridge` | Base learner. For `randomforest`, LOCO-MP and LOCO-AdaMP use decision trees, while LOCO-Split uses random forests. |
| `--loss` | `sq`, `abs` | Squared or absolute loss. |

### Example

```bash
python main.py --dim low --inf coverage --simu independent --ml ridge --loss sq
```

## Case Study

The ROSMAP data used in the case study are not included in this repository. Researchers interested in obtaining the data can consult the [RADC Data Resource Request page](https://www.radc.rush.edu/requests/data.htm) for access requirements and the application process.
