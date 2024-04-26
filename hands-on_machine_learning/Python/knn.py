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


X = df_iris.drop("target", axis=1, inplace=False )
y = df_iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_new = X_test.reset_index(drop=True).iloc[1]


# Find knn (3-nearest neighbors) based on euclidean distance
def euclidian_nearest_neighbors (X: float, X_new: float, k:int) -> float:
    squared_diff= np.square(X - X_new)
    # we dont need to calculare square root since the order is going to be the same.
    distance = np.sum(squared_diff, axis=1)
    top_k_distance = distance.sort_values()[:k]
    return top_k_distance


# Predict output by averaging k nearest neighbors
def knn_predict(top_k: pandas.series , Y: float) -> float:

    pass




squared_diff= np.square(X_train - X_new)
# we dont need to calculare square root since the order is going to be the same.
distance = np.sum(squared_diff, axis=1)
distance_sorted = np.argsort(distance)
top_k = distance.sort_values()[:3]


y_train
top_k_index = top_k.index
y_nearset_k = y_train.loc[top_k_index]

y_nearset_k.value_counts()


randomm = pd.Series(np.random.randint(0, 10, 100))
randomm.value_counts()
randomm.value_counts().argmax()


type(top_k)
X_train["sepal length (cm)"][68]
X_train.iloc[68]
X_train.loc[68]
y_test.

x = np.linspace(0,10,100)
y = np.sin(x)

plt.figure(figsize=(6,10))


plt.subplot(3,1, 1)
plt.scatter(x,y)


plt.subplot(3,1, 2)
plt.plot(x,y)


plt.subplot(3,1, 3)
plt.hist(y, bins=30)
plt.tight_layout()
plt.show()
plt.savefig("plot.png")

df =df_iris.groupby(["target"]).mean().reset_index()

df.columns

Index(['target', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)'],
      dtype='object')

plt.bar(df_iris['target'].values, df_iris['target'].index)

plt.bar( df_iris['target'].index, df_iris['target'].values)
df_iris["target"]


plt.plot(df_iris.index ,df_iris["sepal length (cm)"], df_iris["sepal width (cm)"])
plt.scatter(df_iris["sepal width (cm)"], df_iris["sepal length (cm)"])
plt.bar( df['target'], df['sepal width (cm)'], label='age', color=["red", "blue"])


plt.figure(figsize=(10,6))
plt.hist(df_iris["petal width (cm)"], bins=30, label="petal width (cm)", alpha=0.3)
plt.hist(df_iris["petal length (cm)"], bins=30, label="petal length (cm)", alpha=0.3)
plt.hist(df_iris["sepal width (cm)"], bins=30, label="sepal width (cm)", alpha=0.7)
plt.hist(df_iris["sepal length (cm)"], bins=30, label="sepal length (cm)", alpha=0.3)
plt.legend()


plt.boxplot(df_iris["petal width (cm)"], )