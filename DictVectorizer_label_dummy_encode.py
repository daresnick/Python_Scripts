# LabelEncode and One-Hot encode in one function

# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
dfk_dict = dfk.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
dfk_encoded = dv.fit_transform(dfk_dict)

# Print the resulting first five rows
print(dfk_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)