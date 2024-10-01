# PLS Models with NIPALS Algorithm

## Overview

This repository provides a comprehensive set of tools for developing Partial Least Squares (PLS) models using the NIPALS algorithm. It includes implementations in both MATLAB and Python, allowing users to choose the platform that best fits their needs. The tools support data preprocessing, component selection, model validation, and visualization, along with model inversion functionality for predicting input variables corresponding to a desired output.

## Features

- **Data Preprocessing**: Automatically centers and scales data for robustness. Users can opt-out by adjusting parameters.
- **Component Selection**: Automatically determines the number of components based on the data, with flexibility to specify a different number if needed.
- **Model Framework**: Includes an `alpha` parameter to define the prediction confidence limit, balancing between model accuracy and range.
- **Model Inversion (MI)**: A method for calculating the input (`X_new`) corresponding to a desired output (`Y_des`). Offers two approaches:
  - General PLS Model Inversion (`method=1`).
  - A custom, user-suggested method (`method=2`).
- **Model Outputs**: Provides scores, loadings, Hotelling’s T², Squared Prediction Error (SPE), and additional metrics.
- **Visualization**: Tools for visualizing data distributions, scores, and model performance.

## Implementations

### MATLAB
- **Location**: [matlab/](matlab/)
- **Usage**: Includes scripts for model training, evaluation, and plotting. Refer to the README in the `matlab` folder for specific instructions.

### Python
- **Location**: [python/](python/)
- **Usage**: Provides both class-based and module-based implementations. The README in the `python` folder details the setup and usage.

## Example Usage

### **MATLAB**:
```matlab
% Load your data
X = your_X_data;
Y = your_Y_data;

% Train the PLS model
plsModel = pls_nipals(X, Y, NumComponents, CenterScale);

% Evaluate the model
[T_score, Hoteling_T2, SPE] = pls_evaluation(plsModel, X_new);
pls_ploting(plsModel, X_new);
```

### **Python**:
```python
import numpy as np
from pls_module import pls_nipals
from pls_class import PLSClass

# Load your data
X = np.array(your_X_data)
Y = np.array(your_Y_data)

# Using module
pls_model = pls_nipals(X, Y, Num_com=3, alpha=0.95)
y_pre, T_score, Hoteling_T2, SPE_X, SPE_Y_pre = pls_evaluation(pls_model, X_test)
visual_plot(pls_model, scores_plt, X_test)

# Using class
my_pls_model = PLSClass()
my_pls_model.train(X, Y, Num_com=3, alpha=0.95)
y_pre, T_score, Hoteling_T2, SPE_X, SPE_Y_pre = my_pls_model.evaluation(X_test)
my_pls_model.visual_plot(scores_plt, X_test)

# Model Inversion (MI)
x_des, y_pre_MI = my_pls_model.MI(Y_des=Y_test[1, :].reshape(1, -1), method=1)
print('General MI:', x_des, y_pre_MI)

x_des, y_pre_MI = my_pls_model.MI(Y_des=Y_test[1, :].reshape(1, -1), method=2)
print('Suggested MI:', x_des, y_pre_MI)
```

## Installation

Clone the repository and navigate to the respective `matlab` or `python` directory for setup instructions.

```bash
git clone https://github.com/username/pls_models_nipals.git
cd pls_models_nipals
```

## Acknowledgements

The PLS model implementations are based on established methodologies and have been adapted for ease of use in MATLAB and Python environments. This repository also includes model inversion capabilities that provide flexibility in predicting new input data based on desired outputs.