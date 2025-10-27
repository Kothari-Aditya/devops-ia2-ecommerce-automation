"""
Feature selection module - extract feature importance from trained model.
Extracted from Jupyter notebook.
"""

import pandas as pd


def get_feature_importance(model, feature_names):
    """
    Extract and rank feature importances from XGBoost model.
    Exact logic from notebook (Cell 130).
    
    Args:
        model: Trained XGBoost model
        feature_names: List or Index of feature names
    
    Returns:
        DataFrame with Feature and Importance columns, sorted by importance
    """
    # Exact code from notebook
    xgb_feature_importances = model.feature_importances_
    
    # Create a DataFrame to rank the features
    xgb_feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_feature_importances
    })
    
    # Sort the features by importance
    xgb_feature_importance_df = xgb_feature_importance_df.sort_values(
        by='Importance', 
        ascending=False
    )
    
    return xgb_feature_importance_df


def display_feature_importance(importance_df, top_n=10):
    """
    Display top N most important features.
    
    Args:
        importance_df: DataFrame from get_feature_importance()
        top_n: Number of top features to display
    """
    print(f"\n{'='*60}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print('='*60)
    print(importance_df.head(top_n).to_string(index=False))
    print('='*60)


if __name__ == "__main__":
    print("Feature selector module - use with trained model")
    print("Example usage:")
    print("  importance_df = get_feature_importance(model, X_train.columns)")
    print("  display_feature_importance(importance_df, top_n=10)")