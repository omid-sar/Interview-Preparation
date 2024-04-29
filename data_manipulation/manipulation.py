import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import time

dfo = pd.read_csv('../data/P87-S7-Advanced-Manipulation-Resources/flights2.csv.gz')
print(dfo.columns)
dfo.head(2)

# add double zeros when the time only has minutes.
dfo["SCHEDULED_DEPARTURE"] = dfo["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)
dfo["SCHEDULED_DEPARTURE"] = dfo["SCHEDULED_DEPARTURE"].apply(lambda s: time(hour=int(s[:2]), minute=int(s[-2:])))


df = dfo[["SCHEDULED_DEPARTURE",'AIRLINE','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE','SCHEDULED_ARRIVAL']]
df["DATE"] = pd.to_datetime(dfo[["YEAR", "MONTH","DAY"]])
df.head(3)


df.sort_values("DISTANCE", inplace=True)
df["RANK"] = df["DISTANCE"].rank(method="min")

df.AIRLINE.nunique()
df.AIRLINE.unique()

df.AIRLINE.value_counts()
(df.AIRLINE == "AS").sum()

df.indexcount()

df.AIRLINE.unique().sum()

s = df.AIRLINE.unique()
df_AS = df[df["AIRLINE"] == "AS"]

df['CANCELLATION_REASON'].isna().sum()

df.isna().sum()

dfo.index[-1]
dfo.isna().sum()/(dfo.index[-1]+1)*100


masked_df = df[(df["DATE"] < "2015-01-05") & (df["DATE"] > "2015-01-02") & (df["AIRLINE"] == "AS")]

mask1 = (df["DATE"] < "2015-01-05") & (df["DATE"] > "2015-01-02") & (df["AIRLINE"] == "AS")

mask2 = df["DATE"].between("2015-01-02", "2015-01-05") | (df["AIRLINE"] == "AS")
df.loc[0,["AIRLINE", "DISTANCE"]] # first row and "AIRLINE", "DISTANCE" columns

masked_rows_columns = df.loc[mask1, ["AIRLINE", "DISTANCE"]]


(df == "SEA").any()