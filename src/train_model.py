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


# load our train and test data
with open('data/train_test.pickle', 'rb') as file:
    dataset_list = pickle.load(file)
    
X_train = dataset_list[0]
X_test = dataset_list[1]
y_train = dataset_list[2]
y_test = dataset_list[3]

scoring = 'roc_auc'

# XGBoost Hypertuning
param_grid = {'max_depth': [max_depth for max_depth in range(5,10)],
               'min_child_weight': [min_child_weight for min_child_weight in range(5,10)],
               'eta': [.5, .4, .3, .2, .1, .05, .01, .005],
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
pipe = Pipeline([('classifier' , LogisticRegression())])
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

# save predictions
y_pred_log_reg_train = hypertuned_logreg.predict(X_test)
y_pred_log_reg_test = hypertuned_logreg.predict(X_test)

y_pred_xgboost_train = hypertuned_xgb.predict(X_test)
y_pred_xgboost_test = hypertuned_xgb.predict(X_test)

y_pred_rf_train = hypertuned_rf.predict(X_test)
y_pred_rf_test = hypertuned_rf.predict(X_test)

y_pred_ensemble_train = voting_clf_.predict(X_test)
y_pred_ensemble_test = voting_clf_.predict(X_test)

train_predictions = {'Logistic Regression' : y_pred_log_reg_train,
                     'XGBoost' : y_pred_xgboost_train,
                     'Random Forest' : y_pred_rf_train,
                     'Ensemble' : y_pred_ensemble_train}

test_predictions = {'Logistic Regression' : y_pred_log_reg_test,
                     'XGBoost' : y_pred_xgboost_test,
                     'Random Forest' : y_pred_rf_test,
                     'Ensemble' : y_pred_ensemble_test}

# saving train and test predictions
with open('predictions/train_predictions.pickle', 'wb') as file:
    pickle.dump(train_predictions, file)

with open('predictions/test_predictions.pickle', 'wb') as file:
    pickle.dump(test_predictions, file)























