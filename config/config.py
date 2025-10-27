"""
Configuration file for Delivery Risk Prediction project.
EXACTLY matching the preprocessing from the Jupyter notebook.
"""

import os

# ========== PROJECT PATHS ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained')

# ========== DATA FILES ==========
RAW_DATA_FILE = 'DataCoSupplyChainDataset.csv'  # Exact filename from notebook
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)
MODEL_FILE = 'xgboost_delivery_risk_model.pkl'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
SCALER_FILE = 'minmax_scaler.pkl'
SCALER_PATH = os.path.join(MODELS_DIR, SCALER_FILE)

# ========== MODEL CONFIGURATION ==========
TARGET_COLUMN = 'Late_delivery_risk'
RANDOM_SEED = 42
TEST_SIZE = 0.2

# XGBoost Hyperparameters (as used in notebook)
XGBOOST_PARAMS = {
    'random_state': RANDOM_SEED
}

# ========== COLUMNS TO DROP (EXACT from notebook) ==========
# Stage 1 drops
COLUMNS_TO_DROP_STAGE1 = [
    'Order Zipcode', 'Product Description', 'Customer Id', 'Order Id', 
    'Product Card Id', 'Latitude', 'Longitude', 'Customer Email', 
    'Customer Fname', 'Customer Lname', 'Customer Password', 
    'Customer Street', 'Order Item Cardprod Id', 'Order Customer Id', 
    'Product Image', 'Product Name', 'Category Name', 'Customer State', 
    'Customer Zipcode', 'Department Name'
]

# Stage 2 drops
COLUMNS_TO_DROP_STAGE2 = [
    'Customer Country', 'Customer Segment', 'Market', 'Delivery Status', 
    'Customer City', 'Order City', 'Order Region', 'Order State', 'Order Status'
]

# Stage 3 drops
COLUMNS_TO_DROP_STAGE3 = ['Order Item Discount Rate', 'Product Status']

# Stage 4 drops
COLUMNS_TO_DROP_STAGE4 = ['Order Item Id']

# Final drops (after feature engineering)
COLUMNS_TO_DROP_FINAL = [
    'order date (DateOrders)', 'shipping date (DateOrders)', 
    'ship_day_of_week_name', 'order_day_of_week_name', 
    'ship_daypart', 'order_daypart', 'Order Country'
]

# ========== DATE COLUMNS ==========
SHIPPING_DATE_COL = 'shipping date (DateOrders)'
ORDER_DATE_COL = 'order date (DateOrders)'

# ========== ENCODING ==========
# OneHotEncode this column (using category_encoders)
ONEHOT_COLUMN = 'Type'

# OrdinalEncode Shipping Mode (Cell 104 - has inherent order)
SHIPPING_MODE_COLUMN = 'Shipping Mode'
SHIPPING_MODE_ORDER = ["Standard Class", "Second Class", "First Class", "Same Day"]

# ========== SCALING ==========
# Exact numerical features to scale (from notebook)
NUMERICAL_FEATURES_TO_SCALE = [
    "Days for shipping (real)", 
    "Days for shipment (scheduled)", 
    "Benefit per order", 
    "Sales per customer", 
    "Order Item Discount", 
    "Order Item Product Price", 
    "Order Item Profit Ratio", 
    "Order Item Quantity", 
    "Sales", 
    "Order Item Total", 
    "Order Profit Per Order", 
    "Product Price", 
    "Order_to_Shipment_Time"
]

# ========== DAYPART MAPPING ==========
DAYPART_MAPPING = {
    'Early Morning': 0,
    'Morning': 1,
    'Noon': 2,
    'Eve': 3,
    'Night': 4,
    'Late Night': 5
}