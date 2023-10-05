# 📊 JP-Morgan-Quant-Program

This repository simulates the intricate scenarios and challenges encountered by Quantitative Research (QR) professionals. Each file encapsulates a distinct task, reflecting the diverse roles of the QR team at JPMorgan Chase & Co.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features) 
- [Licence](#licence)

## 🌐 Overview

The JPMorgan Chase & Co Forage Programme amalgamates computational mathematics, statistics, and computer science methodologies to address financial data challenges. This repository encompasses:

1. **Loan Default Prediction** (`loan_default_predictor.py`): Utilises machine learning for loan default predictions, integrating Exploratory Data Analysis (EDA) and RandomForestClassifier.
2. **FICO Score Quantisation** (`fico_quantization.py`): Implements KMeans clustering for FICO score categorisation, enhanced by log-likelihood optimisation and illustrative visualisations.
3. **Dynamic Pricing Modelling** (`pricing_model.py`): A sophisticated pricing model accounting for transactional and storage costs, enriched by dynamic gas pricing strategies.
4. **Gas Price Estimation** (`gas_price_estimator.py`): Forecasts natural gas prices using time-series data via linear regression, complemented by an interactive interface for on-the-fly estimations.

Collectively, these modules provide a robust toolkit, addressing the multifarious challenges inherent in modern finance.

## 🔍 Prerequisites

To run the modules effectively, ensure the following Python libraries are installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

These can be installed using pip: 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 🛠️ Modules

### `loan_default_predictor.py`
- **Data Loading**: Efficient loading and preparation from a specified CSV source.
- **Exploratory Data Analysis (EDA)**:
  - Comprehensive statistical summary.
  - Visualisation of default patterns.
  - Insightful feature correlations.
- **Model**: RandomForestClassifier training and evaluations, including AUC score, ROC curve, and Precision-Recall curve metrics.
- **Feature Importance**: Visual interpretation of feature significance.
- **Loss Estimation**: Predictive financial loss calculations.

### `fico_quantization.py`
- **Data Retrieval**: Effortless extraction from a CSV source.
- **Visual Analysis**:
  - Distribution insights of FICO scores.
  - Demarcation of score thresholds.
- **Quantisation**: KMeans clustering and log-likelihood optimisation for refined bucketing.

### `pricing_model.py`
- **Pricing Computation**:
  - Transactional cost and revenue analysis.
  - Duration-based storage cost evaluations.
- **Detailed Insights**: Comprehensive transactional cost breakdown.
- **Dynamic Pricing**: Date-centric gas price determinations.

### `gas_price_estimator.py`
- **Data Handling**:
  - Transformation from CSV to a Pandas DataFrame.
  - Date adaptation for regression.
- **Time-Series Analysis**: Visual trends of gas prices against time.
- **Linear Regression**: Predictive analytics via scikit-learn's LinearRegression.
- **Interactive Interface**: User-driven predictions with rigorous date validations.

## 📝 Licence

This project is licenced under the [MIT Licence](https://choosealicense.com/licenses/mit/).

Copyright (c) 2023 Joshua Finney

Permission is granted, free of charge, to any individual obtaining a copy of this software and affiliated documentation (the "Software"), to utilise the Software without restriction, encompassing rights such as copying, modifying, merging, publishing, and distribution, whilst also allowing permissions to whom the Software is furnished, given the following stipulations:

The aforementioned copyright notice and this permission notice shall be incorporated in all substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT ANY GUARANTEE OF ANY SORT, BE IT EXPRESS OR IMPLIED, INCLUDING BUT NOT CONFINED TO GUARANTEES OF MERCHANTABILITY, SUITABILITY FOR A SPECIFIC FUNCTION, AND NONINFRINGEMENT. IN NO SCENARIO SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE HELD ACCOUNTABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, RESULTING FROM, OR IN CONNECTION WITH THE SOFTWARE OR ANY UTILISATION OR DEALINGS IN THE SOFTWARE.
