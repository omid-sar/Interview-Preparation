import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfo = pd.read_csv('../data/P87-S7-Advanced-Manipulation-Resources/flights2.csv.gz')
print(dfo.columns)
dfo.head()


dfo.sort_values(["TAIL_NUMBER", "FLIGHT_NUMBER"], inplace=True)
dfo.sort_index(inplace=True)