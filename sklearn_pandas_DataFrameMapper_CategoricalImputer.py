# Use sklearn_pandas Imputers instead!
# Code not run.

# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X_train.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = (X_train.dtypes == object)

# Get list of categorical column names
categorical_columns = X_train.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X_train.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                [([numeric_feature], Imputer(strategy="median")) 
                                for numeric_feature in non_categorical_columns],
                                    input_df=True,
                                    df_out=True
                                    )

# Apply categorical imputer
categorical_imputation_mapper = CategoricalImputer(
                                    [([category_feature], Imputer(strategy='constant')) 
                                    for category_feature in categorical_columns]
                                    )
