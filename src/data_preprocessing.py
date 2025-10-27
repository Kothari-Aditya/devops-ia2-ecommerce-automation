"""
Data preprocessing module - all transformations from notebook.
Follows exact preprocessing logic from Jupyter notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from category_encoders import OneHotEncoder as OHE
from config.config import (
    COLUMNS_TO_DROP_STAGE1, COLUMNS_TO_DROP_STAGE2, 
    COLUMNS_TO_DROP_STAGE3, COLUMNS_TO_DROP_STAGE4,
    COLUMNS_TO_DROP_FINAL, TARGET_COLUMN, TEST_SIZE, RANDOM_SEED,
    SHIPPING_DATE_COL, ORDER_DATE_COL, ONEHOT_COLUMN,
    NUMERICAL_FEATURES_TO_SCALE, DAYPART_MAPPING,
    SHIPPING_MODE_COLUMN, SHIPPING_MODE_ORDER
)


def daypart_function(x):
    """
    Categorize hour into daypart.
    Exact function from notebook (Cell 93).
    """
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Noon'
    elif (x > 16) and (x <= 20):
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return 'Night'
    elif (x <= 4):
        return 'Late Night'


def preprocess_data(data):
    """
    Complete preprocessing pipeline matching notebook exactly.
    
    Steps:
    1. Drop columns (4 stages)
    2. Parse datetime columns
    3. Feature engineering (Order_to_Shipment_Time, day_of_week, hour, daypart)
    4. Drop final columns (dates and categorical versions)
    5. Train-test split
    6. OneHot encoding (Type)
    7. Ordinal encoding (Shipping Mode)
    8. MinMax scaling
    
    Args:
        data: Raw pandas DataFrame
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, ohe, ord_enc
    """
    print("Starting preprocessing...")
    data_cleaned = data.copy()
    
    # ========== STAGE 1: Drop columns (Cell 49) ==========
    print("Stage 1: Dropping columns...")
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE1)
    
    # ========== STAGE 2: Drop more columns (Cell 68) ==========
    print("Stage 2: Dropping more columns...")
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE2)
    
    # ========== STAGE 3: Drop additional columns (Cell 78) ==========
    print("Stage 3: Dropping additional columns...")
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE3)
    
    # ========== STAGE 4: Drop Order Item Id (Cell 82) ==========
    print("Stage 4: Dropping Order Item Id...")
    data_cleaned = data_cleaned.drop(columns=COLUMNS_TO_DROP_STAGE4)
    
    # ========== DATETIME PARSING & FEATURE ENGINEERING ==========
    print("Parsing datetime columns...")
    
    # Convert to datetime (Cell 86)
    data_cleaned[SHIPPING_DATE_COL] = pd.to_datetime(data_cleaned[SHIPPING_DATE_COL])
    data_cleaned[ORDER_DATE_COL] = pd.to_datetime(data_cleaned[ORDER_DATE_COL])
    
    # Create Order_to_Shipment_Time (Cell 86)
    print("Creating Order_to_Shipment_Time feature...")
    data_cleaned['Order_to_Shipment_Time'] = (
        (data_cleaned[SHIPPING_DATE_COL] - data_cleaned[ORDER_DATE_COL])
        .astype('timedelta64[s]') / pd.Timedelta(hours=1)
    ).astype(int)
    
    # Extract day of week (Cell 87, 88)
    print("Extracting day of week features...")
    data_cleaned['ship_day_of_week'] = data_cleaned[SHIPPING_DATE_COL].dt.dayofweek
    data_cleaned['order_day_of_week'] = data_cleaned[ORDER_DATE_COL].dt.dayofweek
    
    # Map day of week to names (Cell 89)
    day_mapping = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    data_cleaned['ship_day_of_week_name'] = data_cleaned['ship_day_of_week'].map(day_mapping)
    data_cleaned['order_day_of_week_name'] = data_cleaned['order_day_of_week'].map(day_mapping)
    
    # Extract hour (Cell 91, 92)
    print("Extracting hour features...")
    data_cleaned['ship_hour'] = data_cleaned[SHIPPING_DATE_COL].dt.hour
    data_cleaned['order_hour'] = data_cleaned[ORDER_DATE_COL].dt.hour
    
    # Create daypart (Cell 94)
    print("Creating daypart features...")
    data_cleaned['ship_daypart'] = data_cleaned['ship_hour'].apply(daypart_function)
    data_cleaned['order_daypart'] = data_cleaned['order_hour'].apply(daypart_function)
    
    # Map daypart to numeric (Cell 95)
    data_cleaned['ship_daypart_n'] = data_cleaned['ship_daypart'].map(DAYPART_MAPPING)
    data_cleaned['order_daypart_n'] = data_cleaned['order_daypart'].map(DAYPART_MAPPING)
    
    # ========== DROP FINAL COLUMNS (Cell 97) ==========
    print("Dropping final columns (dates and categorical versions)...")
    data_cleaned.drop(COLUMNS_TO_DROP_FINAL, axis=1, inplace=True)
    
    # ========== TRAIN-TEST SPLIT (Cell 100) ==========
    print("Splitting into train and test sets...")
    train_set, test_set = train_test_split(
        data_cleaned, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED
    )
    
    X_train = train_set.drop([TARGET_COLUMN], axis='columns')
    y_train = train_set[TARGET_COLUMN]
    
    X_test = test_set.drop([TARGET_COLUMN], axis=1)
    y_test = test_set[TARGET_COLUMN]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # ========== ONE-HOT ENCODING (Cell 106) ==========
    print("One-hot encoding 'Type' column...")
    ohe = OHE(cols=[ONEHOT_COLUMN], use_cat_names=True)
    
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)
    
    # ========== ORDINAL ENCODING (Cell 104) ==========
    print("Ordinal encoding 'Shipping Mode' column...")
    ord_enc_shipping_mode = OrdinalEncoder(categories=[SHIPPING_MODE_ORDER])
    
    X_train[SHIPPING_MODE_COLUMN] = ord_enc_shipping_mode.fit_transform(
        X_train[[SHIPPING_MODE_COLUMN]]
    )
    X_test[SHIPPING_MODE_COLUMN] = ord_enc_shipping_mode.transform(
        X_test[[SHIPPING_MODE_COLUMN]]
    )
    
    # ========== MIN-MAX SCALING (Cell 109) ==========
    print("Scaling numerical features...")
    scaler = MinMaxScaler()
    
    X_train[NUMERICAL_FEATURES_TO_SCALE] = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES_TO_SCALE]
    )
    X_test[NUMERICAL_FEATURES_TO_SCALE] = scaler.transform(
        X_test[NUMERICAL_FEATURES_TO_SCALE]
    )
    
    print("Preprocessing complete!")
    print(f"Final feature count: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler, ohe, ord_enc_shipping_mode


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_data
    
    data = load_data()
    X_train, X_test, y_train, y_test, scaler, ohe, ord_enc = preprocess_data(data)
    
    print("\n=== Preprocessing Test ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")