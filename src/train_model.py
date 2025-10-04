"""
Model training module for the MLOps capstone project.
Handles model training, evaluation, and saving.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data():
    """
    Load processed training and testing data.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        X_train = pd.read_csv('../data/processed/X_train.csv')
        X_test = pd.read_csv('../data/processed/X_test.csv')
        y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
        y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()
        
        logger.info("Processed data loaded successfully")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error(f"Processed data files not found: {e}")
        raise


def train_models(X_train, y_train):
    """
    Train multiple models and return them.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Starting model training...")
    
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} training completed")
    
    return trained_models


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return metrics


def save_model(model, model_name, metrics):
    """
    Save the trained model and its metrics.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model evaluation metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save model
    model_path = f'../models/{model_name}.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = f'../models/{model_name}_metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


def compare_models(models, X_test, y_test):
    """
    Compare multiple models and return the best one.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: Best model name and all metrics
    """
    all_metrics = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
    
    # Find best model based on R² score
    best_model = max(all_metrics, key=lambda x: x['r2'])
    logger.info(f"Best model: {best_model['model_name']} with R² = {best_model['r2']:.4f}")
    
    return best_model['model_name'], all_metrics


def main():
    """Main function to run the training pipeline."""
    try:
        # Load processed data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Train models
        models = train_models(X_train, y_train)
        
        # Evaluate and compare models
        best_model_name, all_metrics = compare_models(models, X_test, y_test)
        
        # Save all models and metrics
        for name, model in models.items():
            model_metrics = next(m for m in all_metrics if m['model_name'] == name)
            save_model(model, name, model_metrics)
        
        # Save comparison results
        import json
        comparison_path = '../models/model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump({
                'best_model': best_model_name,
                'all_metrics': all_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
