# Wine Quality Model - MLOps Project

This project aims to build a **Random Forest Regressor** to predict wine quality using the UCI Wine Quality Dataset. The pipeline is automated using **GitHub Actions**, and the results, including performance metrics and visual plots, are saved and uploaded as artifacts after each run.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Setup Instructions](#setup-instructions)
  - [Running the Model](#running-the-model)
- [How to Modify the Code](#how-to-modify-the-code)
  - [Train-Test Split](#train-test-split)
  - [Model Hyperparameters](#model-hyperparameters)
  - [Feature Importance Plot](#feature-importance-plot)
  - [Residuals Plot](#residuals-plot)
- [GitHub Actions Workflow](#github-actions-workflow)
- [Interpreting Results](#interpreting-results)
- [Contributing](#contributing)

---

## Project Overview

This project builds a machine learning model using **Random Forest** to predict wine quality. After training the model, it calculates the explained variance for both the training and test datasets, saves these metrics to a text file, and generates the following plots:
1. **Feature Importance Plot** – To visualize the relative importance of each feature in the model.
2. **Residuals Plot** – To display the difference between true and predicted wine quality.

The model is automatically trained and evaluated in a **CI/CD** pipeline using GitHub Actions.

---

## Getting Started

### Requirements

To run this project, ensure you have the following:
- Python 3.12 or later
- Libraries listed in `requirements.txt` (e.g., `scikit-learn`, `seaborn`, `matplotlib`, `pandas`, etc.)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdelhadi-essabri/MLOPS.git
   cd MLOPS
   ```

2. **Install the dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the model locally:**
   ```bash
   python train.py
   ```

This will generate the model, calculate metrics, and produce plots stored as PNG files in the `images/` directory.

---

## How to Modify the Code

### 1. Train-Test Split
You can modify the size of the test set in the `train_test_split()` function in `train.py`. For example, to use 30% of the data for testing:

```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=seed)
```

### 2. Model Hyperparameters
To change the hyperparameters of the **Random Forest** model, edit the following line in `train.py`:

```python
regr = RandomForestRegressor(max_depth=5, random_state=seed)
```

For example, if you want to increase the number of trees in the forest, you can modify the code as:

```python
regr = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=seed)
```

### 3. Feature Importance Plot
The feature importance plot is generated using **Seaborn**. To modify its appearance, edit the relevant section in `train.py`. For example, you can adjust the size of the plot by changing:

```python
plt.figure(figsize=(10, 6))  # Change the size as needed
```

### 4. Residuals Plot
The residuals plot shows the difference between true and predicted values. You can add more jitter or change the visual style as needed:

```python
y_pred = regr.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
```

---

## GitHub Actions Workflow

The model can also be trained automatically using **GitHub Actions**. Each time code is pushed to the repository, the pipeline is triggered and performs the following tasks:
1. Installs dependencies.
2. Trains the model.
3. Generates metrics and plots.
4. Uploads the plots as artifacts, which you can download from the GitHub Actions run page.

The workflow is defined in `.github/workflows/cml.yaml`. You can modify it as needed.

---

## Interpreting Results

After running the model, the following files are generated:
- `metrics.txt`: Contains the training and test variance explained.
- `images/feature_importance.png`: Displays the importance of each feature.
- `images/residuals.png`: Shows the relationship between true and predicted wine quality.

These artifacts are also available in the **GitHub Actions** run page after each push.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.
