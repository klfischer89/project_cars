import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv("./data/autos.csv.bz2", encoding = "iso-8859-1")

df = df[df["offerType"] == "Angebot"]
df = df[df["vehicleType"] == "kleinwagen"]
df = df[df["notRepairedDamage"] == "nein"]

df.dropna(inplace = True)