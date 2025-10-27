"""
Model training module - train and evaluate XGBoost model.
Extracted from Jupyter notebook.
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from config.config import XGBOOST_PARAMS, MODEL_PATH, SCALER_PATH, MODELS_DIR


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier.
    Exact code from notebook (Cell 129).
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained XGBoost model
    """
    print("Training XGBoost model...")
    
    # Exact initialization from notebook
    xgb_model = XGBClassifier(**XGBOOST_PARAMS)
    
    # Fit the model
    xgb_model.fit(X_train, y_train)
    
    print("Training complete!")
    return xgb_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    Exact evaluation from notebook (Cell 129).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        y_preds: Predictions on test set
    """
    print("\nEvaluating model...")
    
    # Predict on test set
    y_preds = model.predict(X_test)
    
    # Calculate accuracy (exact from notebook)
    accuracy = accuracy_score(y_test, y_preds)
    print(f"Accuracy: {accuracy}")
    
    # Classification report (exact from notebook)
    report = classification_report(y_test, y_preds)
    print(f"Classification Report:\n {report}")
    
    # Additional: confusion matrix
    cm = confusion_matrix(y_test, y_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return y_preds


def save_model(model, model_path=None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path to save model (default: from config)
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


def save_scaler(scaler, scaler_path=None):
    """
    Save fitted scaler to disk.
    
    Args:
        scaler: Fitted MinMaxScaler
        scaler_path: Path to save scaler (default: from config)
    """
    if scaler_path is None:
        scaler_path = SCALER_PATH
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")


def load_model(model_path=None):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to load model from (default: from config)
    
    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def load_scaler(scaler_path=None):
    """
    Load fitted scaler from disk.
    
    Args:
        scaler_path: Path to load scaler from (default: from config)
    
    Returns:
        Loaded scaler
    """
    if scaler_path is None:
        scaler_path = SCALER_PATH
    
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")
    return scaler


def save_encoder(encoder, encoder_path=None):
    """
    Save fitted encoder to disk.
    
    Args:
        encoder: Fitted OneHotEncoder
        encoder_path: Path to save encoder (default: models/trained/)
    """
    if encoder_path is None:
        encoder_path = os.path.join(MODELS_DIR, 'onehot_encoder.pkl')
    
    joblib.dump(encoder, encoder_path)
    print(f"Encoder saved to: {encoder_path}")


def load_encoder(encoder_path=None):
    """
    Load fitted encoder from disk.
    
    Args:
        encoder_path: Path to load encoder from (default: models/trained/)
    
    Returns:
        Loaded encoder
    """
    if encoder_path is None:
        encoder_path = os.path.join(MODELS_DIR, 'onehot_encoder.pkl')
    
    encoder = joblib.load(encoder_path)
    print(f"Encoder loaded from: {encoder_path}")
    return encoder


def save_ordinal_encoder(ord_encoder, ord_encoder_path=None):
    """
    Save fitted ordinal encoder to disk.
    
    Args:
        ord_encoder: Fitted OrdinalEncoder
        ord_encoder_path: Path to save ordinal encoder (default: models/trained/)
    """
    if ord_encoder_path is None:
        ord_encoder_path = os.path.join(MODELS_DIR, 'ordinal_encoder.pkl')
    
    joblib.dump(ord_encoder, ord_encoder_path)
    print(f"Ordinal encoder saved to: {ord_encoder_path}")


def load_ordinal_encoder(ord_encoder_path=None):
    """
    Load fitted ordinal encoder from disk.
    
    Args:
        ord_encoder_path: Path to load ordinal encoder from (default: models/trained/)
    
    Returns:
        Loaded ordinal encoder
    """
    if ord_encoder_path is None:
        ord_encoder_path = os.path.join(MODELS_DIR, 'ordinal_encoder.pkl')
    
    ord_encoder = joblib.load(ord_encoder_path)
    print(f"Ordinal encoder loaded from: {ord_encoder_path}")
    return ord_encoder


if __name__ == "__main__":
    print("Model trainer module")
    print("Use this module to train, evaluate, and save XGBoost models")