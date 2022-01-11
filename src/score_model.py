from sklearn.metrics import classification_report
import pickle

with open('data/train_test.pickle', 'rb') as file:
    dataset_list = pickle.load(file)
with open('models/train_predictions.pickle', 'rb') as file:
    train_predictions = pickle.load(file)
with open('models/test_predictions.pickle', 'rb') as file:
    test_predictions = pickle.load(file)


# Ground truth
y_test = dataset_list[3]

# Predictions for each model
logreg_train = train_predictions[0]
xgb_train = train_predictions[1]
rf_train = train_predictions[2]
ensemble_train = train_predictions[3]

logreg_test = test_predictions[0]
xgb_test = test_predictions[1]
rf_test = test_predictions[2]
ensemble_test = test_predictions[3]

# Classification report print for each model

def accuracy(pred, test):
    return sum([1 if pred[i] == test[idx] else 0 for i,idx in enumerate(test.index)])/len(test)

print(f"The accuracy using Logistic Regression is {accuracy(logreg_test, y_test)}.")
print(f"The accuracy using XGBoost is {accuracy(xgb_test, y_test)}.")
print(f"The accuracy using Random Forest is is {accuracy(rf_test, y_test)}.")
print(f"The accuracy using the ensemble model is {accuracy(ensemble_test, y_test)}.")