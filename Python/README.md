# PLS Model Development and Evaluation  

This project provides a set of tools for developing Partial Least Squares (PLS) models using the NIPALS algorithm. The PLS model is now available in both a class-based format and a module-based format, allowing users to choose the implementation that best suits their needs. The main functions take input datasets `X` (features) and `Y` (responses) to build a PLS model. The implementation includes options for data preprocessing, component selection, model validation, Null Space (NS) calculations, and comprehensive model outputs.  

## Features  

### Data Preprocessing:  
- By default, the model centers and scales the data to improve the robustness and reliability of the PLS model.  
- Users can bypass the default scaling by setting the `CenterScale` parameter to 0.  

### Component Selection:  
- The number of components is automatically selected as the number of X variables, allowing for monitoring model performance metrics (R-squared) to make a final decision about the required number of components.  
- Users have the flexibility to specify a different number of components if desired.  

### Null Space (NS) Calculations:  
The project includes three functions for Null Space calculations, which allow for identifying the input configurations (`X`) that result in specific predictions (`Y`).  

1. **Null Space for All Columns (`NS_all`)**:  
   Calculates the Null Space for all columns of `Y`, ensuring all `X` data produces the same prediction for all `Y` columns.  
   ```python  
   NS_t, NS_X, NS_Y = MyPlsModel.NS_all(Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
   ```  

2. **Null Space for Single Column (`NS_single`)**:  
   Computes the Null Space for individual columns of `Y`, ensuring all solutions produce the same prediction for the specified `Y` column.  
   ```python  
   NS_t, NS_X, NS_Y = MyPlsModel.NS_single(which_col=1, Num_point=1000, Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
   ```  

3. **Null Space Using `X` Space (`NS_XtoY`)**:  
   Calculates the Null Space directly in the `X` space for individual columns of `Y`, similar to `NS_single`.  
   ```python  
   NS_t, NS_X, NS_Y = MyPlsModel.NS_XtoY(which_col=2, Num_point=1000, Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
   ```  

These functions allow users to explore the input configurations leading to desired outputs, leveraging either the score space or the `X` space.  

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

# Null Space Calculation Examples  
NS_t, NS_X, NS_Y = MyPlsModel.NS_all(Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
NS_t, NS_X, NS_Y = MyPlsModel.NS_single(which_col=1, Num_point=1000, Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
NS_t, NS_X, NS_Y = MyPlsModel.NS_XtoY(which_col=2, Num_point=1000, Y_des=Y[1, :].reshape(1, -1), MI_method=1)  
```  