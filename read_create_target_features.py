# Example of reading in data and creating arrays of features and target for sklearn

import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('file_name.csv')

# Create arrays for features and target variable
y = np.array(df.target_col.values)
X = np.array(df[['feature1', 'feature2', '...']].values)

y = y.reshape(-1,1)
X = X.reshape(-1,1)

