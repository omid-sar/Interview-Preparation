import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris["target"] = iris.target

df_iris

sns.pairplot(df_iris.reset_index(drop=True), hue="target")


X = df_iris.drop("target", axis=1)
y = df_iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_new = X_test.reset_index(drop=True).iloc[0]



# Find knn (3-nearest neighbors) based on euclidean distance
def euclidian_nearest_neighbors (X: float, X_new: float, k:int) -> float:
    squared_diff= np.square(X - X_new)
    # we dont need to calculare square root since the order is going to be the same.
    distance = np.sum(squared_diff, axis=1)
    top_k_distance = distance.sort_values()[:k]
    return top_k_distance


# Predict output by averaging k nearest neighbors
def knn_predict()

np.sqrt(15)

X_train - X_new

(X_train - X_new)*((X_train - X_new).T)

(np.square(X_train - X_new))

np.sum(np.square(X_train - X_new), axis=1, sort=ascending)
np.sort(np.sum(np.square(X_train - X_new), axis=1))

squared_diff= np.square(X_train - X_new)
# we dont need to calculare square root since the order is going to be the same.
distance = np.sum(squared_diff, axis=1)
distance_sorted = np.argsort(distance)
distance.sort_values()[:3]

X_train["sepal length (cm)"][68]
X_train.iloc[68]
X_train.loc[68]
y_test.



data = np.random.randn(100, 6)
np.round(data, 3)
df = pd.DataFrame(data=data, columns=["A", "B", "C", "D", "E", "F"])
df.to_csv("data.csv", float_format="%0.4f")
df2 = pd.read_csv("data.csv")