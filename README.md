# 📊 JP-Morgan-Quant-Program

## 📋 Table of Contents

- [🌐 Overview](#overview)
- [⚙️ Features](#features) 
- [📝 License](#license)

## 🌐 Overview

The JP Morgan Forage Programme merges computational mathematics, statistics, and computer science to tackle financial data challenges. This repository houses specialised modules designed for specific tasks, representing the intricacies of JP Morgan's operations:

1. Loan Default Prediction (loan_default_predictor.py): Uses machine learning to explore and predict loan defaults, employing Exploratory Data Analysis (EDA) and RandomForestClassifier for accurate predictions.

2. FICO Score Quantisation (fico_quantization.py): Categorises FICO scores through KMeans clustering, supplemented with log-likelihood optimisation and visual representation for clarity.

3. Dynamic Pricing Modelling (pricing_model.py): Comprehensive pricing mechanism for transactions, covering transactional and storage costs, underscored by dynamic gas pricing.

4. Gas Price Estimation (gas_price_estimator.py): Predicts natural gas prices using time-series data and linear regression, boasting an interactive user interface for real-time predictions.

Together, these modules form a compact suite of solutions, addressing the diverse challenges of contemporary finance.

## ⚙️ Features

### `loan_default_predictor.py`
- **Comprehensive Data Loading**: Streamlined loading and preparation of dataset from a specified CSV file.
- **Exploratory Data Analysis (EDA)**:
  - Display of summary statistics.
  - Visualization of default distribution.
  - Pairwise feature relationship plots.
  - Heatmap showcasing feature correlations.
- **Data Preprocessing**: Efficient splitting of data into training and test sets.
- **Random Forest Classifier**:
  - Model training using RandomForestClassifier.
  - Evaluation metrics: AUC score, ROC curve, and Precision-Recall curve.
- **Feature Importance Visualization**: Ranking of features based on their impact.
- **Expected Loss Calculation**: Financial loss estimation for a given loan using the trained model.

### `fico_quantization.py`
- **Dynamic Data Loading**: Seamless fetching of dataset from a CSV file.
- **Visual Representation**:
  - FICO score distribution visualization.
  - Highlighting of identified score boundaries.
- **MSE Quantization**: KMeans clustering to bucket FICO scores.
- **Log-Likelihood Optimisation**: Optimal bucket boundary determination by maximising log-likelihood.

### `pricing_model.py`
- **Comprehensive Pricing Model**:
  - Calculation of transaction costs and revenues.
  - Storage cost computation based on transaction durations.
- **Detailed Debugging Information**: Cost breakdown for each transaction.
- **Date-Based Price Series**: Dynamic gas price determination using a date-based dictionary.
- **Sample Runs**: Illustrative application demonstrations.

### `gas_price_estimator.py`
- **Data Parsing & Preprocessing**:
  - Conversion of CSV string data to a Pandas DataFrame.
  - Date transformation for linear regression compatibility.
- **Time-Series Visualisation**: Trend visualisation of natural gas prices against dates.
- **Linear Regression Analysis**:
  - Predictive modeling using scikit-learn's LinearRegression.
  - MSE output for model accuracy assessment.
- **Interactive Price Estimation**:
  - Real-time predictions based on user dates.
  - Robust date format validation.

## 📝 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

Copyright (c) 2023 Joshua Finney

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
