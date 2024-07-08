import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Hello")

def perform_analysis(filename):
    df = pd.read_csv(filename)

    # Ensure the dataframe has 'x' and 'y' columns
    if 'x' not in df.columns or 'y' not in df.columns:
        print("The CSV file must contain 'x' and 'y' columns.")
        return

    # Scatter plot without linear regression
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], color='blue', label='Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot without Linear Regression')
    plt.legend()
    plt.savefig('scatter_plot_without_regression.png')
    plt.show()

    # Scatter plot with linear regression
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], color='blue', label='Data points')

    model = LinearRegression()

    x = np.array(df['x']).reshape(-1, 1)
    y = np.array(df['y'])

    model.fit(x, y)
    r_sq = model.score(x, y)
    print(f"Coefficient of determination (R^2): {r_sq}")

    y_pred = model.predict(x)

    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot with Linear Regression')
    plt.legend()
    plt.savefig('scatter_plot_with_regression.png')
    plt.show()
    
 # Check if at least one command-line argument is provided
if len(sys.argv) < 2:
        print("give me a CSV file!")

filename = sys.argv[1]

    # Check if the file is a CSV
if filename.endswith(".csv"):
     perform_analysis(filename)
else:
     print("The file is not a CSV. Please provide a valid CSV file.")

