# PLS Models with NIPALS Algorithm

## Overview

This project provides a set of tools for developing Partial Least Squares (PLS) models using the NIPALS algorithm. The main function takes input datasets `X` (features) and `Y` (responses) to build a PLS model. The implementation includes options for data preprocessing, component selection, model validation, and comprehensive model outputs.

## Features

- **Data Preprocessing:** 
  - By default, the model centers and scales the data to improve the robustness and reliability of the PLS model.
  - Users can bypass the default scaling by setting the `CenterScale` parameter to `0`.
  
- **Component Selection:**
  - The number of components is automatically selected as the number of X variables allowing to monitor model performance metric (R squared) to make final decision about the required number of Components.
  - Users have the flexibility to specify a different number of components if desired.
  
- **Model Framework:**
  - The function accepts an alpha parameter ranging from 0 to 1 that defines the modeling confidence limit framework within which the model is valid for prediction. A smaller alpha constrains the model's prediction framework but enhances accuracy for new observations. Conversely, a higher alpha allows for a broader range of observations within the model's scope, though it may increase the likelihood of less accurate predictions. 
  
- **Model Outputs:**
  - The function outputs a structure that includes:
    - **Scores and Loadings**: Essential components of the PLS model.
    - **Hotelling’s T²**: A multivariate metric that helps identify outliers.
    - **SPE (Squared Prediction Error)**: Measures the difference between observed and predicted values.
    - **SPE and T² Limits**: Set thresholds for identifying unusual data points.
    - **Scaling and Centering Values**: Retained for future scaling of new data.
    - **Additional Data**: Any other relevant model parameters that may be useful in subsequent analysis.
  
- **Additional Functions:**
  - **Model Evaluation**: Functions that take the PLS model structure and new observation `X` data to calculate model parameters, allowing for the evaluation of new data against the existing model.
  - **Plotting Functions**: Tools to visualize data distributions, score plots, and the SPE and Hotelling T² distributions, aiding in outlier detection and data cleaning.

## Usage

### Model Training

Train a PLS model using the provided function:

```matlab
% Example code to train a PLS model
X = [your_X_data]; % Replace with your X dataset
Y = [your_Y_data]; % Replace with your Y dataset
plsModel = pls_nipals(X, Y, NumComponents(optional),CenterScale(optional));
[T_score,Hoteling T^2, SPE] = pls_evaluation(plsModel,X_new);
[]=pls_ploting(plsModel,X_new(optional))
