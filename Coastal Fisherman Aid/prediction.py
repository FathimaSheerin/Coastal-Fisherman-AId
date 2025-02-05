# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the dataset
df = pd.read_csv("gps data.csv")
df.columns = ['Satellite Count', 'Latitude', 'Longitude']

# Label the border crossing based on Latitude and Longitude
df.loc[df["Latitude"] >= 12.0067, "Border Crossed"] = "yes"
df.loc[df["Longitude"] >= 75.2180, "Border Crossed"] = "yes"
df.loc[df["Latitude"] < 12.0067, "Border Crossed"] = "no"
df.loc[df["Longitude"] < 75.2180, "Border Crossed"] = "no"

# Label Encoding for "Border Crossed"
le = LabelEncoder()
df["Border Crossed"] = le.fit_transform(df["Border Crossed"])

# Features and target variable
X = df[["Latitude", "Longitude"]]
Y = df[["Border Crossed"]]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Training the Decision Tree Regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Predicting on the test data
y_pred = regressor.predict(X_test)

# Reshaping the prediction and converting to list
x = np.reshape(y_pred, -1)
pred_list = x.tolist()

# Count the number of "Illegal Fishing" detections
count = pred_list.count(1.0)

# Determining the prediction
if count > 10:
    prediction = "Illegal Fishing is detected in this Vessel"
else:
    prediction = "No Illegal Fishing is detected in this Vessel"

# Printing the result
print(prediction)

# Saving the trained model using pickle
filename = 'prediction.pkl'
pickle.dump(regressor, open(filename, 'wb'))
