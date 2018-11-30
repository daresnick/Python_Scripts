#Validation function
n_folds = 5

# Inputs: model
# Outputs: rmse values for each split as an array

## One note: Passing the KFold object kf to the cross_val_score() function is a very slick and easy way to use shuffled Kfold
## in your scoring metrics. This method can be used for a lot of different models.

def rmse_cv(model):
'''This function outputs the rmse score for a regression model with shuffled KFold CV'''
    kf = KFold(n_folds, shuffle = True, random_state = 42) # creates the kfold object
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring = "neg_mean_squared_error", cv = kf))
    return(rmse)
