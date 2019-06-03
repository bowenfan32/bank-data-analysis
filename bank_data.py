# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:40:44 2019

@author: Bowen
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing df
df = pd.read_csv('bank-additional-full.csv', sep=';', header=0, index_col=False)

# Visualizations 
sns.set(color_codes=True)
bar_figure = sns.barplot(x=df['age'].unique(), y=df['age'].value_counts(), data=df)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show(bar_figure)

age = sns.distplot(df['age'], bins=40, kde=False);
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show(age)

bar_figure = sns.barplot(x=df['poutcome'], y=df['y'].value_counts(), data=df)
plt.xlabel('Risk Rating')
plt.ylabel('Count')
plt.show(bar_figure)

tes = sns.countplot(x="poutcome", data=df)

loan = sns.countplot(x="loan", data=df)


# Remove biased data
df = df[df.poutcome != 'nonexistent']

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
df['job']      = labelencoder_X.fit_transform(df['job']) 
df['marital']  = labelencoder_X.fit_transform(df['marital']) 
df['education']= labelencoder_X.fit_transform(df['education']) 
df['default']  = labelencoder_X.fit_transform(df['default']) 
df['housing']  = labelencoder_X.fit_transform(df['housing']) 
df['loan']     = labelencoder_X.fit_transform(df['loan']) 
df['contact']     = labelencoder_X.fit_transform(df['contact']) 
df['month']     = labelencoder_X.fit_transform(df['month']) 
df['day_of_week']     = labelencoder_X.fit_transform(df['day_of_week']) 
df['poutcome']     = labelencoder_X.fit_transform(df['poutcome']) 
df['y'] = labelencoder_X.fit_transform(df['y']) 

X = df.iloc[:, 0:20].values
y = df.iloc[:, 20].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


###########################
# Random Forest Classifier
###########################

# Hyeprparameter tuning 

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 100, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=10, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)



from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [34, 38, 42, 46, 50],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3],
    'n_estimators': [120, 130, 140, 150, 160]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 10)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_


# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=120, min_samples_leaf=5, min_samples_split=3, 
                                    max_depth=70, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Scores
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

accuracy_score = accuracy_score(y_test, y_pred)
print('Classification Accuracy: {0:0.3f}'.format(accuracy_score))

precision = precision_score(y_test, y_pred, average=None)
print('Precision score: ', precision)

recall = recall_score(y_test, y_pred, average=None)
print('Recall score: ', recall)


f1 = f1_score(y_test, y_pred)  
print('Average f1 score: {0:0.3f}'.format(f1))

cm = confusion_matrix(y_test, y_pred)
print(cm)



#####################
# XGBoost Classifier
#####################


# Hyperparameter tuning
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5], 'n_estimators':[300,400,500]}
ind_params = {'learning_rate': 0.01, 'seed':0,  
             'objective': 'binary:hinge'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 3, n_jobs = -1, verbose=10) 

optimized_GBM.fit(X_train, y_train)
print (optimized_GBM.best_params_)


feature_names = list(df)
feature_names = feature_names[:-1]
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtrain.feature_names
dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, feature_names=feature_names)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'n_estimator': 300, 'min_child_weight': 5, 'eta': 0.01, 'max_depth': 7,
             'nthread': -1, 'booster' : 'gbtree', 'verbosity': 1,
            'eval_metric': 'rmse', 'objective': 'binary:hinge'}
model = xgb.train(xgb_pars, dtrain, 6000, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)


y_pred_xgb2 = model.predict(dtest)


# Scores
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

accuracy_score = accuracy_score(y_test, y_pred_xgb2)
print('Classification Accuracy: {0:0.3f}'.format(accuracy_score))

precision = precision_score(y_test, y_pred_xgb2, average=None)
print('Precision score: ', precision)

recall = recall_score(y_test, y_pred_xgb2, average=None)
print('Recall score: ', recall)


f1 = f1_score(y_test, y_pred_xgb2)  
print('Average f1 score: {0:0.3f}'.format(f1))

cm = confusion_matrix(y_test, y_pred_xgb2)
print(cm)

from xgboost import plot_importance
plot_importance(model)



