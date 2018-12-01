# Convert the categorical variables to numbers using LabelEncoder

# Create a boolean mask for categorical columns; done to the first 25 columns because they need to have no missing values
categorical_mask = (dfk.dtypes == 'object')

#categorical_columns = ['MSZoning', 'PavedDrive', 'Neighborhood', 'BldgType', 'HouseStyle']

# Get list of categorical column names
categorical_columns = dfk.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(dfk[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
dfk[categorical_columns] = dfk[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(dfk[categorical_columns].head())