import pandas as pd
import generate_features as gf
import train_model as tm
import score_model as sm

#let's create a dummy dataset with 50 rows from our original dataset
df = pd.read_csv('Star99999_raw.csv')
dummy = df.sample(50)
#for a dataset this size, this execution will take around 10 minutes

#we pass it as an argument to generate_features
gf.generate_features(dummy, 'dataframe')
#now we have generated a pickle with a list containing the X_train, y_train, 
# X_test and y_test for the dataset we are using

#we call the function to train our models
tm.train_model()
#now we have generated 4 trained models using Logisitc Regressio, XGBoost, 
#Random Forest and an ensemble of the three

# we call the function to calculate the score
sm.score_model()
#it prints on the screen the classification report for each model and the 
#accuracies

