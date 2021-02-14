import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())

print(train.columns.values)  # print the columns in the dataset


# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)

# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

# remove unwanted columns from the dataset
train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# converting the sex column into integer values of 0 and 1
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], 1).astype(float))  # declaring X with only features and without the outcome
y = np.array(train['Survived'])  # declaring Y with column to be predicted.

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,\
                n_clusters=2, n_init=10,\
    random_state=None, tol=0.0001, verbose=0)  # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)  # feeding the values into the model

# calculating the accuracy of the model with the test dataset.
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("accuracy ", correct/len(X))
