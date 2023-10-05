import io
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA = """
Dates,Prices
10/31/20,1.01E+01
11/30/20,1.03E+01
12/31/20,1.10E+01
1/31/21,1.09E+01
2/28/21,1.09E+01
3/31/21,1.09E+01
4/30/21,1.04E+01
5/31/21,9.84E+00
6/30/21,1.00E+01
7/31/21,1.01E+01
8/31/21,1.03E+01
9/30/21,1.02E+01
10/31/21,1.01E+01
11/30/21,1.12E+01
12/31/21,1.14E+01
1/31/22,1.15E+01
2/28/22,1.18E+01
3/31/22,1.15E+01
4/30/22,1.07E+01
5/31/22,1.07E+01
6/30/22,1.04E+01
7/31/22,1.05E+01
8/31/22,1.04E+01
9/30/22,1.08E+01
10/31/22,1.10E+01
11/30/22,1.16E+01
12/31/22,1.16E+01
1/31/23,1.21E+01
2/28/23,1.17E+01
3/31/23,1.20E+01
4/30/23,1.15E+01
5/31/23,1.12E+01
6/30/23,1.09E+01
7/31/23,1.14E+01
8/31/23,1.11E+01
9/30/23,1.15E+01
10/31/23,1.18E+01
11/30/23,1.22E+01
12/31/23,1.28E+01
1/31/24,1.26E+01
2/29/24,1.24E+01
3/31/24,1.27E+01
4/30/24,1.21E+01
5/31/24,1.14E+01
6/30/24,1.15E+01
7/31/24,1.16E+01
8/31/24,1.15E+01
9/30/24,1.18E+01
"""

def parse_data(data_str: str) -> pd.DataFrame:
    """Parse CSV string into DataFrame and convert 'Dates' column to datetime format."""
    df = pd.read_csv(io.StringIO(data_str), parse_dates=['Dates'])
    return df

def plot_data(df: pd.DataFrame) -> None:
    """Plot prices against dates."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Dates'], df['Prices'])
    plt.title('Natural Gas Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def train_regression_model(df: pd.DataFrame) -> LinearRegression:
    """Train a regression model using the given data."""
    df['Dates'] = df['Dates'].apply(lambda x: x.toordinal())
    
    X = df['Dates'].values.reshape(-1, 1)
    y = df['Prices']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Training completed. Mean Squared Error: {mse:.4f}")
    
    return model

def estimate_price(model: LinearRegression, date_str: str) -> float:
    """Estimate the price based on a trained model and a given date string."""
    try:
        date_obj = datetime.strptime(date_str, '%m/%d/%y')
        date_ordinal = date_obj.toordinal()
        return model.predict([[date_ordinal]])[0]
    except ValueError:
        print("Invalid date format. Please enter the date in the format: mm/dd/yy.")
        return None

def main():
    print("Loading and parsing data...")
    df = parse_data(DATA)
    
    print("Plotting the natural gas prices over time...")
    plot_data(df)

    print("Training the regression model...")
    model = train_regression_model(df)
    
    while True:
        date_input = input("\nEnter the date (format: mm/dd/yy) to estimate the gas price or 'exit' to quit: ")
        
        if date_input.lower() == 'exit':
            break
        
        price = estimate_price(model, date_input)
        if price:
            print(f"Estimated price on {date_input}: ${price:.2f}")

if __name__ == '__main__':
    main()