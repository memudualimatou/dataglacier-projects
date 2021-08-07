import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

data = pd.read_csv('Admission_Predict_Ver1.1.csv')
data = data.drop(columns=['Serial No.'])
print(data)

# checking for missing data
print(data.isnull().sum())

# rename some columns
Data = data.rename(columns={'GRE Score': 'GRE',
                            'TOEFL Score': 'TOEFL',
                            'University Rating': 'university_Rating',
                            'Chance of Admit ': 'Chance_of_Admit'},
                   inplace=True)

# remove some outliers
data = data[data.Chance_of_Admit > 0.40]
data.reset_index(drop=True, inplace=True)

# splitting the data
x = data.iloc[:, :-1].values
y = data.iloc[:, 7].values

# split dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Fitting linear regression Regression to the dataset
Lin_regressor = LinearRegression(normalize=True)
Lin_regressor.fit(X_train, y_train)

# To save the model to the disk (serialization) for future use
pickle.dump(Lin_regressor, open('admission_model.pkl', 'wb'))

y_pred = Lin_regressor.predict(X_test)
y_train_pred = Lin_regressor.predict(X_train)

print("RMSE score on the test set:", sqrt(mean_squared_error(y_test, y_pred)))
print("RMSE score on the training set:", sqrt(mean_squared_error(y_train, y_train_pred)))

print("R2 score on the test set:", r2_score(y_test, y_pred) * 100)
print("R2 score on the training set:", r2_score(y_train, y_train_pred) * 100)

# cross validation
# Applying k-Fold Cross Validation USED TO JUST IMPROVE THE MODEL PERFORMANCE(ACCURACY)

accuracies = cross_val_score(estimator=Lin_regressor, X=X_train, y=y_train, cv=5)
print(accuracies.mean(), accuracies.std())


