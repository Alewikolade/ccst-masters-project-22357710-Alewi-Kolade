# Fraud Detection in Finance: A Comprehensive Supervised Learning Approach with Simulated Financial Transaction Data

**Kolade Alewi**  
Student Number: 223057710  

**Supervised by: Dr. Ben Derrick**  

University of the West of England  
August 2024  

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Models and Evaluation](#models-and-evaluation)
5. [Requirements](#requirements)
6. [Usage](#usage)
7. [Acknowledgments](#acknowledgments)
8. [Abstract](#abstract)

## Project Overview

This project aims to develop a robust machine learning model to detect financial fraud within financial transactions. The model addresses challenges such as data imbalance and evolving fraud tactics to effectively identify and prevent fraudulent transactions. This project utilizes machine learning algorithms including XGBoost, Logistic Regression, and Random Forest to build a comprehensive fraud detection system.

## Dataset

The dataset used for this project is a synthetic financial dataset generated using the PaySim simulator, which is available on Kaggle. The dataset contains 6,362,620 transactions with 11 variables including transaction type, amount, account balances, and flags for fraud detection.

### Key Features:
- **step**: Maps a unit of time in the real world. 
- **type**: Type of transaction (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- **amount**: Amount of the transaction.
- **isFraud**: Identifies a fraudulent transaction.
- **isFlaggedFraud**: Flags potentially fraudulent transactions.

### Download and Save the Dataset

To ensure the dataset is accessible from the present working directory:

1. **Download the Dataset**: Visit [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and download the PaySim synthetic financial dataset.
2. **Save the Dataset**: Place the downloaded dataset file (`PS_20174392719_1491204439457_log.csv`) in the root directory of this project (the present working directory).

## Methodology

### Data Preprocessing
1. **Data Cleaning**: Checked for missing values and handled zero balances in accounts.
2. **Feature Engineering**: Created new features to capture discrepancies in account balances.
3. **Data Balancing**: Addressed class imbalance using SMOTE for oversampling and Random Undersampling for Logistic Regression.

### Machine Learning Model Development
Three models were developed:
1. **XGBoost**: Used for its robustness and ability to handle imbalanced datasets.
2. **Logistic Regression**: Employed for its simplicity and effectiveness in binary classification.
3. **Random Forest**: Chosen for its ability to handle large datasets and prevent overfitting.

### Evaluation
Models were evaluated using:
- **Precision, Recall, and F1-Score**: To measure the accuracy of the models in detecting fraud.
- **PR AUC Score**: To evaluate the models' performance on imbalanced datasets.
- **Confusion Matrix**: To visualize the models' prediction errors and accuracy.

## Models and Evaluation

### XGBoost
- **Precision**: Improved to 0.95 after tuning.
- **Recall**: Adjusted to 0.87, indicating effective detection of fraud cases.
- **PR AUC Score**: 0.96567.

### Logistic Regression
- **Precision**: Increased to 0.75 after tuning.
- **Recall**: Decreased to 0.47, showing less effectiveness in detecting fraud.
- **PR AUC Score**: 0.578035.

### Random Forest
- **Precision**: Improved to 0.90 after tuning.
- **Recall**: Adjusted to 0.90, showing balanced performance.
- **PR AUC Score**: 0.966101.

## Requirements

To run the notebook and reproduce the results, the following Python packages are required:

```
imbalanced-learn==0.12.3
scikit-learn==1.3.2
xgboost==2.1.1
matplotlib==3.7.1
numpy==1.26.4
pandas==2.1.4
seaborn==0.13.1
```

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Ensure Dataset is in the Present Working Directory**: Before running the notebook, make sure the dataset (`PS_20174392719_1491204439457_log.csv`) is saved in the same directory where the notebook is located.

2. **Run the Jupyter Notebook**: Open the `Fraud_detection.ipynb` notebook and run all cells sequentially to execute the data preprocessing, model training, and evaluation steps.

## Acknowledgments

I would like to express my sincere gratitude to Dr. Ben Derrick for his invaluable guidance and support throughout this project. I also extend my appreciation to my colleagues for their feedback and suggestions during model development and my family for their unwavering support throughout this journey.

## Abstract

This project explores a supervised learning approach to detect financial fraud using a simulated dataset of financial transactions. By employing machine learning models such as XGBoost, Logistic Regression, and Random Forest, the study addresses challenges like data imbalance and evolving fraud tactics. The research demonstrates that models like XGBoost and Random Forest outperform traditional methods, offering a balance between high precision and recall in fraud detection. The findings underscore the need for adaptive, continuously updated models to effectively identify and prevent fraudulent transactions in dynamic financial environments.
