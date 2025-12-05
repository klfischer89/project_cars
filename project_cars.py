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

df.dropna(inplace = True)

X = df[["kilometer", "yearOfRegistration", "brand"]]
y = df["price"]

cf = ColumnTransformer([
    ("brand", OneHotEncoder(), ["brand"])
], remainder = "passthrough")
 
cf.fit(X)

X_transformed = cf.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size = 0.75)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

X_pred = pd.DataFrame([
    [150000, 2000, "audi"],
    [150000, 2000, "bmw"]
], columns = ["kilometer", "yearOfRegistration", "brand"])

print(model.predict(cf.transform(X_pred)))

