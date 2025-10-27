"""
Prediction module - make predictions on new data.
Uses trained model and applies same preprocessing as training.
"""

import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder as OHE
from config.config import (
    COLUMNS_TO_DROP_STAGE1, COLUMNS_TO_DROP_STAGE2,
    COLUMNS_TO_DROP_STAGE3, COLUMNS_TO_DROP_STAGE4,
    COLUMNS_TO_DROP_FINAL, SHIPPING_DATE_COL, ORDER_DATE_COL,
    ONEHOT_COLUMN, NUMERICAL_FEATURES_TO_SCALE, DAYPART_MAPPING,
    SHIPPING_MODE_COLUMN, SHIPPING_MODE_ORDER
)
from src.data_preprocessing import daypart_function
from src.model_trainer import load_model, load_scaler, load_encoder, load_ordinal_encoder


def preprocess_new_data(data, encoder, scaler, ord_encoder):
    """
    Apply same preprocessing as training data.
    Must match the exact steps from data_preprocessing.py.
    
    Args:
        data: Raw DataFrame with new data
        encoder: Fitted OneHotEncoder (from training)
        scaler: Fitted MinMaxScaler (from training)
        ord_encoder: Fitted OrdinalEncoder (from training)
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    print("Preprocessing new data...")
    data_cleaned = data.copy()
    
    # ========== SAME COLUMN DROPS AS TRAINING ==========
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE1, errors='ignore')
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE2, errors='ignore')
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE3, errors='ignore')
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE4, errors='ignore')
    
    # ========== SAME DATETIME PARSING & FEATURE ENGINEERING ==========
    data_cleaned[SHIPPING_DATE_COL] = pd.to_datetime(data_cleaned[SHIPPING_DATE_COL])
    data_cleaned[ORDER_DATE_COL] = pd.to_datetime(data_cleaned[ORDER_DATE_COL])
    
    # Order_to_Shipment_Time
    data_cleaned['Order_to_Shipment_Time'] = (
        (data_cleaned[SHIPPING_DATE_COL] - data_cleaned[ORDER_DATE_COL])
        .astype('timedelta64[s]') / pd.Timedelta(hours=1)
    ).astype(int)
    
    # Day of week
    data_cleaned['ship_day_of_week'] = data_cleaned[SHIPPING_DATE_COL].dt.dayofweek
    data_cleaned['order_day_of_week'] = data_cleaned[ORDER_DATE_COL].dt.dayofweek
    
    # Day of week names
    day_mapping = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    data_cleaned['ship_day_of_week_name'] = data_cleaned['ship_day_of_week'].map(day_mapping)
    data_cleaned['order_day_of_week_name'] = data_cleaned['order_day_of_week'].map(day_mapping)
    
    # Hour
    data_cleaned['ship_hour'] = data_cleaned[SHIPPING_DATE_COL].dt.hour
    data_cleaned['order_hour'] = data_cleaned[ORDER_DATE_COL].dt.hour
    
    # Daypart
    data_cleaned['ship_daypart'] = data_cleaned['ship_hour'].apply(daypart_function)
    data_cleaned['order_daypart'] = data_cleaned['order_hour'].apply(daypart_function)
    
    # Daypart numeric
    data_cleaned['ship_daypart_n'] = data_cleaned['ship_daypart'].map(DAYPART_MAPPING)
    data_cleaned['order_daypart_n'] = data_cleaned['order_daypart'].map(DAYPART_MAPPING)
    
    # Drop final columns
    data_cleaned.drop(COLUMNS_TO_DROP_FINAL, axis=1, inplace=True, errors='ignore')
    
    # Drop target if present (for prediction, we don't need it)
    if 'Late_delivery_risk' in data_cleaned.columns:
        data_cleaned = data_cleaned.drop(['Late_delivery_risk'], axis=1)
    
    # ========== SAME ENCODING & SCALING ==========
    # OneHot encoding
    data_cleaned = encoder.transform(data_cleaned)
    
    # Ordinal encoding (Shipping Mode)
    data_cleaned[SHIPPING_MODE_COLUMN] = ord_encoder.transform(
        data_cleaned[[SHIPPING_MODE_COLUMN]]
    )
    
    # MinMax scaling
    data_cleaned[NUMERICAL_FEATURES_TO_SCALE] = scaler.transform(
        data_cleaned[NUMERICAL_FEATURES_TO_SCALE]
    )
    
    print("Preprocessing complete!")
    return data_cleaned


def predict(model, X_new):
    """
    Make predictions on new data.
    
    Args:
        model: Trained XGBoost model
        X_new: Preprocessed features
    
    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Probability scores for each class
    """
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    return predictions, probabilities


def predict_new_data(data_path, model_path=None, scaler_path=None):
    """
    End-to-end prediction pipeline for new data.
    
    Args:
        data_path: Path to new CSV data
        model_path: Path to trained model (default: from config)
        scaler_path: Path to fitted scaler (default: from config)
    
    Returns:
        DataFrame with original data + predictions + probabilities
    """
    # Load model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Load new data
    print(f"Loading new data from: {data_path}")
    data = pd.read_csv(data_path, encoding='ISO-8859-1')
    print(f"Loaded {len(data)} rows")
    
    # Note: We need the encoder too, but it's not saved separately in the notebook
    # For prediction, we'll need to save the encoder during training
    print("Warning: Encoder must be loaded/provided separately")
    
    # Preprocess
    # X_new = preprocess_new_data(data, encoder, scaler)
    
    # Predict
    # predictions, probabilities = predict(model, X_new)
    
    # Combine results
    # results = data.copy()
    # results['Predicted_Late_Delivery'] = predictions
    # results['Probability_On_Time'] = probabilities[:, 0]
    # results['Probability_Late'] = probabilities[:, 1]
    
    # return results
    
    print("Note: Complete prediction requires encoder from training")
    return None


if __name__ == "__main__":
    print("Predictor module")
    print("Use this module to make predictions on new data")
    print("Requires trained model, scaler, and encoder")