import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Data preprocess:
data = pd.read_csv('joined_dataframe.csv')
data = data.drop(['Num_Acc', 'Total_grav'], axis=1)
data = data.replace('[^\d.]', '', regex=True).astype(float)  # fix strings
scaler = MinMaxScaler()
X = scaler.fit_transform(data.drop(['num_of_inj'], axis=1))
y = data['num_of_inj'].apply(lambda x: x if x < 5 else 5)  # 10 classes of output (1-5, 5+)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest:
model = RandomForestClassifier()
model.fit(X_train, y_train)
print('Random forest classifier accuracy:')
print(model.score(X_test, y_test))

# KNN:
model = KNeighborsRegressor()
model.fit(X_train, y_train)
print('KNN classifier accuracy:')
print(model.score(X_test, y_test))

# Logistic Regression:
print('Logistic regression classifier accuracy:')
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# SVM:
model = svm.SVC()
model.fit(X_train, y_train)
print('Support vector machine classifier accuracy:')
print(model.score(X_test, y_test))