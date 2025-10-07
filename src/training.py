# src/training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load dataset (from DVC tracked location)
data = pd.read_csv("data/raw/boston_housing.csv")

# Simple preprocessing
X = data.drop("MEDV", axis=1)  # assuming target column is 'MEDV'
y = data["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment
mlflow.set_experiment("boston_housing_experiment")

with mlflow.start_run():
    # Parameters
    learning_rate = 0.01
    epochs = 100
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Log model artifact
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"Training done. MSE: {mse}")
