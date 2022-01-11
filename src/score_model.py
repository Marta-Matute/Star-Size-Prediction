from sklearn.metrics import classification_report
import pickle

def score_model():
    with open('data/train_test.pickle', 'rb') as file:
        dataset_list = pickle.load(file)
        
    X_test = dataset_list[1]
    y_test = dataset_list[3]
        
    with open('models/hypertuned_xgb.pickle', 'rb') as file:
        hypertuned_xgb = pickle.load(file)
    with open('models/hypertuned_rf.pickle', 'rb') as file:
        hypertuned_rf = pickle.load(file)
    with open('models/hypertuned_logreg.pickle', 'rb') as file:
        hypertuned_logreg = pickle.load(file)
    with open('models/voting_clf.pickle', 'rb') as file:
        voting_clf = pickle.load(file)
    
    model_names = ['Logistic Regression', 'XGBoost', 'Random Forest', 'Ensemble']
    
    
    # save predictions
    y_pred_log_reg_train = hypertuned_logreg.predict(X_test)
    y_pred_log_reg_test = hypertuned_logreg.predict(X_test)
    
    y_pred_xgboost_train = hypertuned_xgb.predict(X_test)
    y_pred_xgboost_test = hypertuned_xgb.predict(X_test)
    
    y_pred_rf_train = hypertuned_rf.predict(X_test)
    y_pred_rf_test = hypertuned_rf.predict(X_test)
    
    y_pred_ensemble_train = voting_clf.predict(X_test)
    y_pred_ensemble_test = voting_clf.predict(X_test)
    
    train_predictions = {'Logistic Regression' : y_pred_log_reg_train,
                         'XGBoost' : y_pred_xgboost_train,
                         'Random Forest' : y_pred_rf_train,
                         'Ensemble' : y_pred_ensemble_train}
    
    test_predictions = {'Logistic Regression' : y_pred_log_reg_test,
                         'XGBoost' : y_pred_xgboost_test,
                         'Random Forest' : y_pred_rf_test,
                         'Ensemble' : y_pred_ensemble_test}
    
    
    # Classification report print for each model
    for model in model_names:
        print(f"The classification report for {model} is:")
        print(classification_report(test_predictions[model], y_test))
    
    def accuracy(pred, test):
        return sum([1 if pred[i] == test[idx] else 0 for i,idx in enumerate(test.index)])/len(test)
    
    print("To see the accuracy with more precision:")
    for model in model_names:
        print(f"The accuracy using {model} is {accuracy(test_predictions[model], y_test)}.")