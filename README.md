# Coastal Fisherman Aid: Illegal Fishing Detection System

## Project Overview

The **Coastal Fisherman Aid** project aims to provide a real-time detection system for identifying potential illegal fishing activities using GPS data (Latitude and Longitude). The system uses machine learning techniques to predict whether a vessel has crossed the border into restricted waters, indicating illegal fishing.

Key features of the project include:
- **GPS Data Analysis**: The system analyzes GPS coordinates to predict whether a vessel has crossed the border.
- **Decision Tree Classifier**: A Decision Tree Regressor model is used to classify GPS data and predict illegal fishing activities.
- **Real-Time Alerts**: If illegal fishing is detected, the system provides an alert.
- **Flask Web Interface**: A web interface is created using Flask to input Latitude and Longitude values and display predictions.
  
The project provides an easy-to-use interface that allows users to input the location of a vessel and check if illegal fishing is occurring in real-time.

## Key Features
- **Machine Learning Model**: Uses Decision Tree Regressor for illegal fishing detection based on GPS coordinates.
- **Web Interface**: Built with Flask, users can input vessel GPS data and get predictions for illegal fishing activities.
- **Real-Time Prediction**: Instant alerts on whether the vessel is engaged in illegal fishing or not.

## Technologies Used
- **Python**: The core language for developing the machine learning model and web application.
- **Flask**: For building the web application that hosts the machine learning model.
- **Scikit-learn**: To implement the Decision Tree Regressor for prediction.
- **Pandas**: For data processing and manipulation.
- **NumPy**: For numerical computations.
- **Pickle**: To save the trained model for future predictions.
- **HTML/CSS**: For designing the web interface.

## Installation Guide

To run this project locally, follow these steps:

### Step 1: Clone the repository
```bash
git clone https://github.com/FathimaSheerin/Coastal-Fisherman-Aid.git
cd Coastal-Fisherman-Aid
```

### Step 2: Install the required dependencies
Create a virtual environment and activate it, then install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Flask Application
Once the dependencies are installed, you can run the Flask application:
```bash
python app.py
```

This will start the web server. Navigate to `http://127.0.0.1:5000/` in your browser to access the application.

## How to Use the Application
1. Open the web interface.
2. Input the **Latitude** and **Longitude** of the vessel.
3. Click on **Predict** to check if illegal fishing is detected.

The system will then display whether illegal fishing is occurring based on the GPS coordinates provided.

## Model Training
The model is trained using GPS data with Latitude and Longitude values. A decision tree regressor is used to predict if the vessel has crossed into restricted waters based on its GPS coordinates.

### Model Details:
- **Input Features**: Latitude, Longitude.
- **Target Feature**: Border Crossed (yes/no).
- **Model Type**: Decision Tree Regressor.

The trained model is saved as a pickle file `prediction.pkl` to be used in the Flask application for predictions.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and create a pull request.

## License
This project is open-source and available under the [MIT License](LICENSE).
