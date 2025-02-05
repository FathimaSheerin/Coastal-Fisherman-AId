# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Load the trained model
filename = 'prediction.pkl'
regressor = pickle.load(open(filename, 'rb'))

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle the prediction based on user input"""
    if request.method == 'POST':
        # Get Latitude and Longitude from form input
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])
        
        # Prepare the input data and make the prediction
        data = np.array([[Latitude, Longitude]])
        my_prediction = regressor.predict(data)
        
        # Render the result page with the prediction
        return render_template('result.html', prediction=int(my_prediction[0]))

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
