import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("./data/autos.csv.bz2", encoding = "iso-8859-1")

df = df[df["offerType"] == "Angebot"]
df = df[df["vehicleType"] == "kleinwagen"]
df = df[df["notRepairedDamage"] == "nein"]
df = df[(df["fuelType"] == "benzin") | (df["fuelType"] == "diesel") | (df["fuelType"] == "hybrid")]
df.dropna(inplace = True)

X = df[["kilometer", "yearOfRegistration", "brand", "gearbox", "fuelType"]]

cf = ColumnTransformer([
    ("brand", OneHotEncoder(drop = "first"), ["brand"]),
    ("gearbox", OneHotEncoder(drop = "first"), ["gearbox"]),
    ("fuelType", OneHotEncoder(drop = "first"), ["fuelType"])
], remainder = "passthrough")
 
cf.fit(X)

X_transformed = cf.transform(X)

y = df["price"]

scores = []
for i in range(0, 1000):

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size = 0.75)

    model = LinearRegression()
    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))
    
print(np.mean(scores))

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size = 0.75)

model = LinearRegression()
model.fit(X_train, y_train)

X_pred = pd.DataFrame([
    [150000, 2000, "bmw", "automatik", "benzin"]
], columns = ["kilometer", "yearOfRegistration", "brand", "gearbox", "fuelType"])

print(model.predict(cf.transform(X_pred)))