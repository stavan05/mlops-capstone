"""
Data preprocessing module for the MLOps capstone project.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info("Starting data cleaning...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        # For this example, we'll drop rows with missing values
        df = df.dropna()
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    final_rows = len(df)
    if initial_rows != final_rows:
        logger.info(f"Removed {initial_rows - final_rows} duplicate rows")
    
    logger.info("Data cleaning completed")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    logger.info("Starting feature engineering...")
    
    # Create a copy to avoid modifying the original
    df_eng = df.copy()
    
    # Example: Create a new feature (room-to-people ratio)
    if 'RM' in df_eng.columns and 'PTRATIO' in df_eng.columns:
        df_eng['ROOM_TO_PEOPLE_RATIO'] = df_eng['RM'] / df_eng['PTRATIO']
        logger.info("Created ROOM_TO_PEOPLE_RATIO feature")
    
    # Example: Create log-transformed features for skewed variables
    skewed_features = ['CRIM', 'NOX', 'LSTAT']
    for feature in skewed_features:
        if feature in df_eng.columns:
            df_eng[f'{feature}_LOG'] = np.log1p(df_eng[feature])
            logger.info(f"Created {feature}_LOG feature")
    
    logger.info("Feature engineering completed")
    return df_eng


def prepare_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for training by splitting and scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preparing data for training...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    logger.info(f"Data split completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    """Main function to demonstrate the preprocessing pipeline."""
    # Load data
    data = load_data('../data/raw/boston_housing.csv')
    
    # Clean data
    cleaned_data = clean_data(data)
    
    # Feature engineering
    engineered_data = feature_engineering(cleaned_data)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        engineered_data, target_column='MEDV'
    )
    
    # Save processed data
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)
    
    logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
