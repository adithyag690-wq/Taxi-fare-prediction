# Taxi-fare-prediction
Taxi Fare Prediction using Scikit-learn is a machine learning project that predicts taxi trip fares based on trip distance, passenger count, pickup hour, and whether the trip occurs on a weekend. The project uses a Random Forest Regressor and provides both command-line predictions and a Streamlit-based web interface.
# Taxi Fare Prediction using Scikit-learn

## Project Overview

Taxi Fare Prediction is a machine learning project that estimates the fare of a taxi trip using important trip details such as distance, passenger count, pickup time, and weekend status.

The project uses a RandomForestRegressor model from Scikit-learn to train on taxi trip data and predict the estimated fare for new trips.

## Features

* Predicts taxi fares using machine learning.
* Uses trip distance, passenger count, pickup hour, and weekend status as input features.
* Uses Random Forest Regression for prediction.
* Evaluates the model using MAE, RMSE, and R² score.
* Generates a feature importance chart.
* Provides a command-line prediction script.
* Includes a Streamlit web application for interactive fare estimation.
* Includes a sample taxi fare dataset.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Project Structure

text
Taxi_Fare_Prediction_Sklearn/
│
├── data/
│   └── taxi_fares.csv
│
├── models/
│   └── taxi_fare_model.pkl
│
├── outputs/
│   └── feature_importance.png
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── HOW_TO_RUN.md
└── README.md


## Input Features

The model uses the following features:

* trip_distance_km - Distance travelled in kilometers.
* passenger_count - Number of passengers.
* pickup_hour - Pickup time in 24-hour format.
* is_weekend - Indicates whether the trip occurs on a weekend.

## Target

* fare_amount - Estimated taxi fare.

## Installation

Install the required Python libraries:

bash
pip install -r requirements.txt


## Train the Model

Run:

bash
python train_model.py


This trains the Random Forest model, evaluates its performance, saves the trained model, and generates the feature importance chart.

## Make a Prediction

Run:

bash
python predict.py


The script uses sample trip information and displays the predicted taxi fare.

## Run the Web Application

To launch the Streamlit application:

bash
streamlit run app.py


The application allows users to enter trip details and interactively calculate an estimated taxi fare.

## Model Evaluation

The project evaluates the trained model using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

## Purpose

This project demonstrates how machine learning regression techniques can be used to estimate taxi fares and build a simple interactive prediction application.

## Conclusion

The Taxi Fare Prediction project provides a practical example of using Scikit-learn and Random Forest Regression for fare estimation. It covers data processing, model training, evaluation, prediction, visualization, and deployment through a Streamlit interface.
