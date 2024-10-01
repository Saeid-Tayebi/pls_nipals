# PLS Model Development and Evaluation

This project provides a set of tools for developing Partial Least Squares (PLS) models using the NIPALS algorithm. The PLS model is now available in both a class-based format and a module-based format, allowing users to choose the implementation that best suits their needs. The main functions take input datasets `X` (features) and `Y` (responses) to build a PLS model. The implementation includes options for data preprocessing, component selection, model validation, and comprehensive model outputs.

## Features

### Data Preprocessing:
- By default, the model centers and scales the data to improve the robustness and reliability of the PLS model.
- Users can bypass the default scaling by setting the `CenterScale` parameter to 0.

### Component Selection:
- The number of components is automatically selected as the number of X variables, allowing for monitoring model performance metrics (R-squared) to make a final decision about the required number of components.
- Users have the flexibility to specify a different number of components if desired.

### Model Framework:
- The function accepts an `alpha` parameter ranging from 0 to 1 that defines the modeling confidence limit framework within which the model is valid for prediction. A smaller `alpha` constrains the model's prediction framework but enhances accuracy for new observations. Conversely, a higher `alpha` allows for a broader range of observations within the model's scope, though it may increase the likelihood of less accurate predictions.

### Model Inversion (MI):
- The **`MI`** method allows users to calculate the corresponding `X_new` based on a desired `Y_des`. This is useful for determining the necessary input values to achieve a given output.
  - **Method 1**: Implements the general PLS Model Inversion solution.
  - **Method 2**: Uses a suggested alternative inversion method for potentially improved performance.

### Model Outputs:
- The function outputs a structure that includes:
  - **Scores and Loadings:** Essential components of the PLS model.
  - **Hotelling’s T²:** A multivariate metric that helps identify outliers.
  - **SPE (Squared Prediction Error):** Measures the difference between observed and predicted values.
  - **SPE and T² Limits:** Set thresholds for identifying unusual data points.
  - **Scaling and Centering Values:** Retained for future scaling of new data.
  - **Additional Data:** Any other relevant model parameters that may be useful in subsequent analysis.

### Additional Functions:
- **Model Evaluation:** Functions/methods that take the PLS model structure and new observation X data to calculate model parameters, allowing for the evaluation of new data against the existing model.
- **Plotting Functions:** Tools to visualize data distributions, score plots, and the SPE and Hotelling T² distributions, aiding in outlier detection and data cleaning.

## Advantages
- The code allows for visual representation of both the training data and the new data to be evaluated.
- It is easy to use and treats the developed PLS model like a class carrying all important information, which can then be utilized in other parts of the code.

## Usage

### Set Parameters
```python
X = your_X_data  # Replace with your X dataset
Y = your_Y_data  # Replace with your Y dataset

Num_com = 3       # Number of PLS components (=Number of X Variables)
alpha = 0.95      # Confidence limit (=0.95)

X_test = np.array([[0.9, 0.1, 0.2], [0.5, 0.4, 0.9]])  # New observation data
scores_plt = np.array([1, 2])  # Scores for plotting
```

### Model Implementation as a Module
```python
# Import the PLS module
import pls_module as pls_m

# Call the PLS NIPALS function
pls_model = pls_m.pls_nipals(X, Y, Num_com, alpha)

# Validate the model with new or testing observation data
y_pre, T_score, Hotelin_T2, SPE_X, SPE_Y_pre = pls_m.pls_evaluation(pls_model, X_test)
print(f'Y_pre={y_pre}\n', f'T_score={T_score}\n', f'Hotelin_T2={Hotelin_T2}\n', f'SPE_X={SPE_X}\n', f'SPE_Y_pre={SPE_Y_pre}\n')

# Visualize the data distributions using the module
pls_m.visual_plot(pls_model, scores_plt, X_test, True, True)
```

### Model Implementation as a Class
```python
# Import the PLS class
from pls_class import PLSClass as pls_c

# Create an instance of the PLS class and train the model
MyPlsModel = pls_c()
MyPlsModel.train(X, Y, Num_com, alpha)

# Validate the model with new or testing observation data
y_pre, T_score, Hotelin_T2, SPE_X, SPE_Y_pre = MyPlsModel.evaluation(X_test)
MyPlsModel.visual_plot(scores_plt, X_test)

print(f'Y_pre={y_pre}\n', f'T_score={T_score}\n', f'Hotelin_T2={Hotelin_T2}\n', f'SPE_X={SPE_X}\n', f'SPE_Y_pre={SPE_Y_pre}\n')

# Model Inversion (MI) Example
Y_test = your_Y_test_data  # Replace with your Y test data

# Apply Model Inversion to obtain X_new for a desired Y
x_des, y_pre_MI = MyPlsModel.MI(Y_des=Y_test[1, :].reshape(1, -1), method=1)
print('Original MI:', x_des, y_pre_MI, Y_test[1, :])

x_des, y_pre_MI = MyPlsModel.MI(Y_des=Y_test[1, :].reshape(1, -1), method=2)
print('Suggested MI:', x_des, y_pre_MI, Y_test[1, :])
```