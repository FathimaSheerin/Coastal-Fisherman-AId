# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# Load the Random Forest CLassifier model
filename = 'prediction.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Latitude= float(request.form['Latitude'])
        Longitude= float(request.form['Longitude'])
        #regressor = DecisionTreeRegressor()
        data = np.array([[Latitude,Longitude]])
        my_prediction = regressor.predict(data)
        return render_template('result.html', prediction=int(my_prediction[0])) 

if __name__ == '__main__':
	app.run(debug=True)