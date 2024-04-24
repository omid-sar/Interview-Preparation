import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
df.head()

X = df["GDP per capita (USD)"].values
Y = df["Life satisfaction"].values

# Finf knn based on euclidean distance

def 

