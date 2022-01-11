import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings


#models
from sklearn.linear_model import LogisticRegression

# ensemble models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

def train_model(dataset_list=None):
    # load our train and test data
    if(dataset_list == None):
        with open('data/train_test.pickle', 'rb') as file:
            dataset_list = pickle.load(file)
        
    X_train = dataset_list[0]
    y_train = dataset_list[2]
    
    scoring = 'roc_auc'
    
    print("Training the models")
    
    # XGBoost Hypertuning
    param_grid = {'max_depth': [max_depth for max_depth in range(5,10)],
                   'min_child_weight': [min_child_weight for min_child_weight in range(5,10)],
                   'eta': [0.05, 0.01, 0.005],
                  }
    
    xgbc = GridSearchCV(estimator = XGBClassifier(use_label_encoder=False,eval_metric='auc', objective='reg:squarederror'), param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1, scoring=scoring)
    hypertuned_xgb = xgbc.fit(X_train, y_train)
    
    # Random Forest Hypertuning
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_split': [2, 7],
                   'min_samples_leaf': [1, 3],
                   'bootstrap': [True, False]
                  }
    
    rf_random = GridSearchCV(estimator = RandomForestClassifier(), param_grid = random_grid, cv = 5, verbose=2, n_jobs=-1, scoring=scoring)
    hypertuned_rf = rf_random.fit(X_train, y_train)
    
    # Logistic Regression Hypertuning
    pipe = Pipeline([('classifier' , LogisticRegression(max_iter=500))])
    param_grid = [
        {'classifier__C' : np.logspace(-4, 4, 20),
         'classifier__fit_intercept' : [True, False],
         'classifier__class_weight' : [dict, 'balanced'],
         'classifier__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga']}
    ]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1, scoring=scoring)
    
    hypertuned_logreg = clf.fit(X_train, y_train)
    
    # Ensemble model creation
    estimators = [('xgb', hypertuned_xgb), ('rf', hypertuned_rf), ('lr', hypertuned_logreg)]#, ('knn', hypertuned_knn)]
    voting_clf_ = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf_.fit(X_train, y_train)
    
    #save models
    with open('models/hypertuned_xgb.pickle', 'wb') as file:
        pickle.dump(hypertuned_xgb, file)
    with open('models/hypertuned_rf.pickle', 'wb') as file:
        pickle.dump(hypertuned_rf, file)
    with open('models/hypertuned_logreg.pickle', 'wb') as file:
        pickle.dump(hypertuned_logreg, file)
    with open('models/voting_clf.pickle', 'wb') as file:
        pickle.dump(voting_clf_, file)
    
    























