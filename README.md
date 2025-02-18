# PLS Models with NIPALS Algorithm

## Overview

This repository provides a comprehensive implementation of Partial Least Squares (PLS) regression using the NIPALS algorithm. The project is designed for ease of use, offering advanced features such as model inversion (`xpredict`), null space determination, plotting, and evaluation of new data points. The implementation is validated against the `sklearn` PLS model, ensuring accuracy while providing additional functionality.

## Key Features

- **NIPALS Algorithm**: Implements the PLS regression using the NIPALS algorithm for robust and efficient modeling.
- **Advanced Functionality**:
  - **Model Inversion (`xpredict`)**: Predict input variables (`X`) corresponding to a desired output (`Y`).
  - **Null Space Determination**: Explore the null space of the PLS model for advanced analysis.
  - **Visualization**: Plotting tools for scores, Hotelling’s T², and SPE (Squared Prediction Error).
  - **Evaluation**: Evaluate new data points with confidence limits and metrics.
- **Compatibility**: Validated against the `sklearn` PLS model for consistency.
- **Tests**: A comprehensive `tests` folder to validate different aspects of the PLS implementation.
- **Ease of Use**: Designed with user-friendly methods for fitting, predicting, and evaluating models.

## Project Structure

```
project/
├── pls_nipals/                     # Main PLS implementation
│   ├── pls.py               # PLSClass implementation
│   ├── lib/                 # Supporting libraries (e.g., PCA)
├── tests/                   # Test cases for PLS functionality
│   ├── test_pls.py          # Unit tests for PLSClass
│   └── refrence_model/      # Reference implementation (e.g.,sklearn PLS)
│   └── PLS_MATLAB/          # MATLAB implementation (if needed)
├── requirements.txt         # Python dependencies
├── LICENSE                  # Project license
└── README.md                # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Saeid-Tayebi/pls_nipals.git
   cd pls_models_nipals
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. The package can also be downloaded and installed from the [Releases](https://github.com/Saeid-Tayebi/pls_nipals/releases/tag/pls_nipals_first_release) section.

## Usage

### Basic Usage

```python
import numpy as np
from pls.pls import PlsClass

# Generate sample data
X = np.random.rand(30, 4)  # 30 samples, 4 features
Beta = np.random.rand(4, 2) * 2 - 1  # Random coefficients
Y = X @ Beta  # Target variable

# Fit the PLS model
pls_model = PlsClass()
pls_model.fit(X, Y, n_component=2)

# Predict new data
X_test = np.random.rand(5, 4)  # 5 new samples
eval_result = pls_model.evaluation(X_test)
print("Predicted Y:", eval_result.yfit)
print("score values:", eval_result.tscore)
print("Hotelling’s T²:", eval_result.HT2)
print("SPE:", eval_result.spex)
```

### Model Inversion (`xpredict`)

```python
# Predict X for a desired Y
Y_des = np.array([[1.5, 2.0]])  # Desired output
x_pred = pls_model.x_predict(Y_des, method=1)
print("Predicted X:", x_pred)
```

### Null Space Determination

```python
# Explore the null space for a desired Y
Y_des = np.array([[1.5, 2.0]])  # Desired output
NS_t, NS_X, NS_Y = pls_model.null_space_all(Y_des=Y_des)
print("Null Space X:", NS_X)
print("Null Space Y:", NS_Y)
```

### Visualization

```python
# Visualize the PLS model
pls_model.visual_plot(score_axis=[1, 2], X_test=X_test, color_code_data=None, data_labeling=True)
```

## Tests

The `tests` folder contains unit tests to validate the PLS implementation. You can run the tests using `pytest`:

```bash
pytest tests/
```

The tests cover:

- Model fitting and prediction.
- Model inversion (`xpredict`).
- Null space determination.
- Data preprocessing (e.g., handling zero-variance rows/columns).

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Acknowledgements

This implementation is based on established PLS methodologies and has been extended with additional features for ease of use and advanced analysis. Special thanks to the `sklearn` team for providing a reference implementation.
