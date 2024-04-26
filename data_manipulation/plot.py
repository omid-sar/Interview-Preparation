import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


iris = load_iris()
df= pd.DataFrame(iris.data, columns=['sepal_len', 'sepal_width', 'petal_len','petal_width'])
df["target"] = iris.target


x = np.linspace(0, 10, 100)
y = np.sin(x)


##-------------------------------------------------------------------##
# LINE PLOT(connect the points like a continuouse plot)
plt.plot(x,y)
plt.title("SIN")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

##-------------------------------------------------------------------##
# SCATTER PLOT & LINE PLOT (SEPERATE)
plt.figure(figsize=(10,6))
plt.subplot(2, 1, 1)
plt.scatter(x,y)
plt.title("SCATTER SIN PLOT")
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(2, 1, 2)
plt.plot(x,y)
plt.tight_layout()
plt.show();


# SCATTER PLOT & LINE PLOT (MERGED)
plt.figure(figsize=(10,6))
plt.scatter(x,y, color="red", label="SCATTER")
plt.plot(x,y, label="Line Plot")
plt.legend();


##-------------------------------------------------------------------##
# BAR PLOT(Need at least X-axis parameter and Y-axis parameter)
print(df.head());


plt.figure(figsize=(10,6))
plt.bar(df["target"], df['sepal_len'], label="sepal_len", alpha=1)
plt.bar(df["target"], df['sepal_width'], label="sepal_width", alpha=0.3)
plt.legend();


##-------------------------------------------------------------------##
# HISTOGRAM (ONLY ONE parametr is needeed X-axis is distribution 
## and Y-axis is magnitiude)


plt.figure(figsize=(10,6))
plt.hist(df['sepal_len'], label="sepal_len", bins=30, alpha=0.6)
plt.hist (df['sepal_width'], label="sepal_width", bins=30 ,alpha=0.5)
plt.hist(df['petal_len'], label="petal_len", bins=30, alpha=0.6)
plt.hist (df['petal_width'], label="petal_width", bins=30, alpha=0.5)
plt.legend();



##-------------------------------------------------------------------##
# BOX PLOT


df_box_plot = df.drop('target', axis=1)

plt.figure(figsize=(10,6))
plt.boxplot(df_box_plot, labels=df_box_plot.columns)
plt.title('Boxplot for Each Column in DataFrame')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()


##-------------------------------------------------------------------##
# VI PLOT


plt.figure(figsize=(10,6))
plt.violinplot(df_box_plot, )
plt.title('Boxplot for Each Column in DataFrame')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()


##-------------------------------------------------------------------##
# HEATMAP 

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True);
