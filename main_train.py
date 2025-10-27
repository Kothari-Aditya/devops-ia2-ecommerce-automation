"""
Main training pipeline orchestrator.
Bands together all modules to train the XGBoost model.

Usage:
    python main_train.py
"""

import sys
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.model_trainer import (
    train_xgboost, evaluate_model, save_model, save_scaler, 
    save_encoder, save_ordinal_encoder
)
from src.feature_selector import get_feature_importance, display_feature_importance


def main():
    """
    End-to-end training pipeline.
    
    Steps:
    1. Load raw data
    2. Preprocess data (exact notebook steps)
    3. Train XGBoost model
    4. Evaluate model
    5. Display feature importance
    6. Save model, scaler, and encoder
    """
    print("="*70)
    print("DELIVERY RISK PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    try:
        # ========== STEP 1: Load Data ==========
        print("\n[STEP 1] Loading data...")
        data = load_data()
        
        # ========== STEP 2: Preprocess Data ==========
        print("\n[STEP 2] Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler, encoder, ord_enc = preprocess_data(data)
        
        # ========== STEP 3: Train Model ==========
        print("\n[STEP 3] Training XGBoost model...")
        model = train_xgboost(X_train, y_train)
        
        # ========== STEP 4: Evaluate Model ==========
        print("\n[STEP 4] Evaluating model...")
        y_preds = evaluate_model(model, X_test, y_test)
        
        # ========== STEP 5: Feature Importance ==========
        print("\n[STEP 5] Extracting feature importance...")
        importance_df = get_feature_importance(model, X_train.columns)
        display_feature_importance(importance_df, top_n=15)
        
        # ========== STEP 6: Save Everything ==========
        print("\n[STEP 6] Saving model, scaler, and encoders...")
        save_model(model)
        save_scaler(scaler)
        save_encoder(encoder)
        save_ordinal_encoder(ord_enc)
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nSaved files:")
        print("  - Model: models/trained/xgboost_delivery_risk_model.pkl")
        print("  - Scaler: models/trained/minmax_scaler.pkl")
        print("  - OneHot Encoder: models/trained/onehot_encoder.pkl")
        print("  - Ordinal Encoder: models/trained/ordinal_encoder.pkl")
        print("\nYou can now use main_predict.py to make predictions!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you have placed 'DataCoSupplyChainDataset.csv' in data/raw/")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()