from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold

    
def CrossValidate(dffull, model, splits = 5, seed = 100):
   '''
	Function that applies k-fold CV to a chosen regression model
	Inputs are:
    dffull = data frame with the target in the first column and the features in the other columns
    model = this is the model that it will fit then predict on, example: lr = linear_model.LinearRegression() 
    splits = integer number of folds
    seed = integer seed value for the KFold

	Ouputs are:
    Model
    Average Intercept and Coefficients values or Feature Importance for each feature
    Average R^2 value
    All MAE values
    Average MSE value
    Average MWSE value
    std MSE value
    Average RMSE value
    Average MAE value
	'''
	
    y = dffull.iloc[:,0].values # target column values, removes column names
    X = dffull.iloc[:,1:].values # feature columns values, removes column names

    cv_object = KFold(n_splits = splits, shuffle = True, random_state = seed)
    #cv_object = RepeatedKFold(n_splits = splits, n_repeats = 10, random_state = seed)
    
    # Initialize values to sum
    tot_Coeff = 0
    tot_MSE = 0
    tot_R2 = 0
    count = 0
    tot_feature_importance = 0
    tot_N = 0
    tot_WMSE = 0
    tot_MAE = 0
    
    MSE_all = []
    WMSE_all = []
    MAE_all = []
    
    tot_var = 0
    
    for train_indices, test_indices in cv_object.split(X,y): 
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #the code below is for linear models only
        if (type(model) == type(linear_model.LinearRegression())):
            Coeff = np.concatenate((np.array([lr.intercept_]), lr.coef_), axis = 0)

            tot_Coeff = tot_Coeff + Coeff
            
        #the code below is for ranodom forest models only
        elif (type(model) == type(RandomForestRegressor(n_estimators=20, oob_score=True, random_state=0))):
            #feature_importance
            feature_importance = model.feature_importances_ 
                      
            tot_feature_importance = tot_feature_importance + feature_importance
            
            # Calculate variance of predictions using forestsci
            #import forestci as fci
            #var_unbiased = fci.random_forest_error(model, X_train, X_test)
            #avg_var_f = np.mean(var_unbiased)
            
            #tot_var = tot_var + avg_var_f
                    
        #the code below is for gradient boosting regression models only
        elif (type(model) == type(ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4, 
                    min_samples_split = 2, learning_rate = 0.01, loss ='ls'))):
            #feature_importance
            feature_importance = model.feature_importances_ 
                      
            tot_feature_importance = tot_feature_importance + feature_importance
        
        else:
            print("No model selected")
            break
                        
        # MSE score: 0 is perfect prediction
        MSE = mean_squared_error(y_test,y_pred)
        MSE_all.append(MSE) # Save all MSE's to an array        
        
        # MSE score: 0 is perfect prediction
        MAE = mean_absolute_error(y_test,y_pred)
        MAE_all.append(MSE) # Save all MSE's to an array
        
        # Size of each fold
        n_k = len(y_test)
        tot_N = tot_N + n_k
        
        # Weight MSE's
        WMSE = n_k * MSE
        WMSE_all.append(WMSE) # Save all WMSE's to an array
        
        # R^2 score: 1 is perfect prediction
        R2 = r2_score(y_test, y_pred)
           
        tot_MSE = tot_MSE + MSE
        tot_R2 = tot_R2 + R2
        tot_WMSE = tot_WMSE + WMSE
        tot_MAE = tot_MAE + MAE
            
        #length of loop
        count = count + 1
    
    # The code below return the output as such:
    # model: type of model, avg_Coeff/Feature Importance: average of intercepts/coefficients/feature importance
    # avg_R2: average R squared, MSE_all: all K MSEs, avg_MSE: average of all K MSEs

    # Averages that are the same for all types of models
    avg_MSE = tot_MSE/count
    avg_R2 = tot_R2/count
    avg_WMSE = tot_WMSE/tot_N
    std_MAE = np.std(MAE_all)
    avg_RMSE = avg_MSE**(0.5)
    avg_MAE = tot_MAE/count
    WMSE_dbN_all = [x/tot_N for x in WMSE_all] # WMSE_all/tot_N, but they are in a list
    
    # Average LR
    if (type(model) == type(linear_model.LinearRegression())):
        avg_Coeff = tot_Coeff/count
        return (model, avg_Coeff, avg_R2, MAE_all, avg_MSE, avg_WMSE, std_MAE, avg_RMSE, avg_MAE)  
    
    # Average RFR
    elif (type(model) == type(RandomForestRegressor(n_estimators = 20, oob_score = True, random_state = 0))):
        avg_feature_importance = tot_feature_importance/count
        avg_var = np.mean(tot_var/count)
        return (model, avg_feature_importance, avg_R2, MAE_all, avg_MSE, avg_WMSE, std_MAE, avg_RMSE, avg_MAE)
   
    # Average GBR
    elif (type(model) == type(ensemble.GradientBoostingRegressor(n_estimators = 20, max_depth = 4, 
                    min_samples_split = 2, learning_rate = 0.01, loss ='ls'))):
        avg_feature_importance = tot_feature_importance/count
        return (model, avg_feature_importance, avg_R2, MAE_all, avg_MSE, avg_WMSE, std_MAE, avg_RMSE, avg_MAE)
    
    else:
        print("Something went wrong")