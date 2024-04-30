import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris["target"] = iris.target

sns.pairplot(df_iris.reset_index(drop=True), hue="target")

df_iris["target"].unique() # 3 different categorize
X = df_iris.drop("target", axis=1, inplace=False )
y = df_iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_new = X_test.reset_index(drop=True).iloc[1]


# Find knn (3-nearest neighbors) based on euclidean distance
def euclidian_nearest_neighbors (X: float, y: int,  X_new: float, k:int) -> float:
    squared_diff= np.square(X - X_new)
    # we dont need to calculare square root since the order is going to be the same.
    distance = np.sum(squared_diff, axis=1)
    top_k_distance = distance.sort_values()[:k]
    top_k_class = list(y.loc[top_k_distance.index])
    return top_k_class

top_k_class = euclidian_nearest_neighbors(X_train, y_train, X_new, k=3)


# Predict output by averaging k nearest neighbors
def knn_predict(top_k_class: int , Y: float) -> int:
    

    pass


