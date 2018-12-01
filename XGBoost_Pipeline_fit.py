# Take out target from train set
X_train, y_train = dfk.iloc[:,:-1], dfk.iloc[:,-1]

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline using xgb!
dfk_xgbfit = xgb_pipeline.fit(X_train.to_dict("records"), y_train)
