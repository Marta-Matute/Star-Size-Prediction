import pandas as pd
import re
import math
import os
import pickle
from sklearn.model_selection import train_test_split

# dataset download
url = 'https://github.com/Marta-Matute/Star-Size-Prediction/raw/main/datasets/Star99999_raw.csv'
df = pd.read_csv(url)

#droping missing values
df.dropna(inplace = True)

#float conversion for column 'Vmag'
df['Vmag'] = df['Vmag'].astype(float)

#outlier removal for column 'Vmag'
idx_to_remove = [idx for idx in df.index if df['Vmag'].loc[idx] < 0]
df.drop(idx_to_remove, inplace = True)

#remove spaces in strings in columns 'Plx', 'e_Plx' and 'B-V'
df['Plx'] = [value.replace(' ', '') for value in df['Plx']]
df['e_Plx'] = [value.replace(' ', '') for value in df['e_Plx']]
df['B-V'] = [value.replace(' ', '') for value in df['B-V']]

#remove rows with empty strings in the column 'Plx' and e_Plx
idx_to_remove = [idx for idx in df.index if df['Plx'][idx] == '']
idx_to_remove += [idx for idx in df.index if df['e_Plx'][idx] == '']
idx_to_remove += [idx for idx in df.index if df['B-V'][idx] == '']
df.drop(idx_to_remove, inplace=True)

#float conversion for columns 'Plx', 'e_Plx' and 'B-V'
df['Plx'] = df['Plx'].astype(float)
df['e_Plx'] = df['e_Plx'].astype(float)
df['B-V'] = df['B-V'].astype(float)

#removal for negative values in 'Plx' and 'B-V'
idx_to_remove = [idx for idx in df.index if df['Plx'].loc[idx] <= 0]
idx_to_remove += [idx for idx in df.index if df['B-V'].loc[idx] < 0 or df['B-V'].loc[idx] > 2]
df.drop(idx_to_remove, inplace = True)
    

def outlier_ranges(df, variable):
    '''
    Parameters
    ----------
    df : pandas Dataframe
        Dataframe from which to decide outliers.
    variable : string
        Name of the column we want to find the outliers for.

    Returns
    -------
    lower_bound : float
        lower bound below which a value in the data should be considered outlier
    upper_bound : float
        upper bound above which a value in the data should be considered outlier.
    '''
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_bound = df[variable].quantile(0.25) - 3 * IQR
    upper_bound = df[variable].quantile(0.75) + 3 * IQR
    return (lower_bound, upper_bound)

#removing rows outside the interquantile range for e_Plx
lb, ub = outlier_ranges(df, 'e_Plx')
idx_to_remove = [idx for idx in df.index if df['e_Plx'][idx] < lb or df['e_Plx'][idx] > ub]
df.drop(idx_to_remove, inplace=True)

# TARGET ATTRIBUTE COLUMN CREATION
#target column for our dataset
Giant = []

#list of indexes corresponding to giant, dwarfs, special and mixed stars respectively
giant_idx = []
dwarf_idx = []
special_idx = []
mixed_idx = []

for idx,sptypes in zip(df.index, df['SpType']):
    sptype_list = re.findall(r"(IV|VI|V|III|II|I)", sptypes)

    giant_b = False
    dwarf_b = False
    
    for sptype in sptype_list:
        if(sptype == "I" or sptype == "II" or sptype == "III"): giant_b = True
        elif(sptype == "IV" or sptype == "V" or sptype == "VI"): dwarf_b = True
        
    if(giant_b and dwarf_b): mixed_idx.append(idx); Giant.append(-1)
    elif(giant_b): giant_idx.append(idx); Giant.append(1)
    elif(dwarf_b): dwarf_idx.append(idx); Giant.append(0)
    else: special_idx.append(idx); Giant.append(-1)

# adding the new column with binary values: 1 for giant, 0 for dwarf
df['Giant'] = Giant 

#droping the instances for special and mixed stars
df.drop(special_idx, inplace = True)
df.drop(mixed_idx, inplace = True)

#droping the column containing the spectral type
df.drop('SpType', axis = 1, inplace = True)

#new column, with the value of the absolute magnitude of the star
Amag = []
for vmag,plx in zip(df['Vmag'], df['Plx']):
    M = vmag + 5*math.log(plx+1,10)
    Amag.append(M)

#adding the new column to the dateset
df['Amag'] = Amag

#rearranging the columns of the dataset
new_cols = ['Vmag', 'Plx', 'e_Plx', 'Amag', 'B-V', 'Giant']
df = df[new_cols]

#normalizing the values of the columns (except the target one) using 0 to 1 and min-max methods
cols_to_normalize = ['Vmag', 'Plx', 'e_Plx', 'Amag', 'B-V']
normalized_df = (df[cols_to_normalize]-df[cols_to_normalize].mean())/df[cols_to_normalize].std()
min_max_normalized_df = (df[cols_to_normalize]-df[cols_to_normalize].min())/(df[cols_to_normalize].max()-df[cols_to_normalize].min())

#adding target variable to the normalized datasets
normalized_df['Giant'] = df['Giant']
min_max_normalized_df['Giant'] = df['Giant']

# set predictors
X = normalized_df.drop(['Giant'], axis=1)
X_min_max = min_max_normalized_df.drop(['Giant'], axis=1)
# set variable to predict
y = normalized_df['Giant']
y_min_max = min_max_normalized_df['Giant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_min_max, X_test_min_max, y_train_min_max, y_test_min_max = train_test_split(X_min_max, y_min_max, test_size=0.2)

#gathering all the different datasets in a list
dataset_list = [X_train, X_test,
                y_train, y_test]

dataset_list_min_max = [X_train_min_max, X_test_min_max,
                        y_train_min_max, y_test_min_max]

#creating the directory where the dataset will be stored
if not os.path.isdir('data'):
    os.mkdir('data')

#saving the list in a pickle
with open('data/train_test.pickle', 'wb') as file:
    pickle.dump(dataset_list, file)

with open('data/train_test_min_max.pickle', 'wb') as file:
    pickle.dump(dataset_list_min_max, file)

