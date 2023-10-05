# JP-Morgan-Quant-Program


# Features

## `loan_default_predictor.py`
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

## `fico_quantization.py`
- **Dynamic Data Loading**: Seamless fetching of dataset from a CSV file.
- **Visual Representation**:
  - FICO score distribution visualization.
  - Highlighting of identified score boundaries.
- **MSE Quantization**: KMeans clustering to bucket FICO scores.
- **Log-Likelihood Optimization**: Optimal bucket boundary determination by maximizing log-likelihood.

## `pricing_model.py`
- **Comprehensive Pricing Model**:
  - Calculation of transaction costs and revenues.
  - Storage cost computation based on transaction durations.
- **Detailed Debugging Information**: Cost breakdown for each transaction.
- **Date-Based Price Series**: Dynamic gas price determination using a date-based dictionary.
- **Sample Runs**: Illustrative application demonstrations.

## `gas_price_estimator.py`
- **Data Parsing & Preprocessing**:
  - Conversion of CSV string data to a pandas DataFrame.
  - Date transformation for linear regression compatibility.
- **Time-Series Visualization**: Trend visualization of natural gas prices against dates.
- **Linear Regression Analysis**:
  - Predictive modeling using scikit-learn's LinearRegression.
  - MSE output for model accuracy assessment.
- **Interactive Price Estimation**:
  - Real-time predictions based on user dates.
  - Robust date format validation.
