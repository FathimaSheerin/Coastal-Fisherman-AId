# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import pickle


df = pd.read_csv("gps data.csv")
df.columns =['Satellite Count', 'Latitude', 'Longitude']
df.loc[df["Latitude"]>=12.0067,"Border Crossed"]="yes"
df.loc[df["Longitude"]>=75.2180,"Border Crossed"]="yes"
df.loc[df["Latitude"]<12.0067,"Border Crossed"]="no"
df.loc[df["Longitude"]<75.2180,"Border Crossed"]="no"
le = LabelEncoder()
df["Border Crossed"] = le.fit_transform(df["Border Crossed"])
X = df[["Latitude","Longitude"]]
Y = df[["Border Crossed"]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
x = np.reshape(y_pred,-1)
list = x.tolist()
count = list.count(1.0)
if count>10:
    prediction="Illegal Fishing is detected in this Vessel"
else:
    prediction="No Illegal Fishing is detected in this Vessel"
print(prediction)

# Creating a pickle file for the classifier
filename = 'prediction.pkl'
pickle.dump(regressor, open(filename, 'wb'))