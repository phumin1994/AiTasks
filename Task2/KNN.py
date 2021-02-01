import pandas as pd
from sklearn.model_selection import train_test_split
# Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  # dataset link
# Specifying column names.
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)  # loading the dataset
print(iris)

# Convert the species into Integer Value
iris_class = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['species_num'] = [iris_class[i] for i in iris.species]

# Create an 'X' matrix by dropping the irrelevant columns.
X = iris.drop(['species', 'species_num'], axis=1)
y = iris.species_num


# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Instantiate the model with 5 neighbors.
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model
knn.fit(X_train, y_train)
# Print the accuracy with Y_test
print(knn.score(X_test, y_test) * 100, '%')  # this is the accuracy of model, it is a simple dataset so it is 100%.

# press cntrl+shift+f10 to run