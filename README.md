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
