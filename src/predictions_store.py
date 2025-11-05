"""
Predictions Storage Module
Handles saving and retrieving predictions from JSON file storage
Used by DevOps pipeline for batch processing
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Storage configuration
STORAGE_DIR = Path(__file__).parent.parent / 'data'
STORAGE_FILE = STORAGE_DIR / 'predictions.json'

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_predictions():
    """
    Load predictions from JSON file
    
    Returns:
        dict: Predictions data structure
    """
    if not STORAGE_FILE.exists():
        return {'predictions': []}
    
    try:
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[STORAGE] Warning: Corrupted storage file, creating new one")
        return {'predictions': []}
    except Exception as e:
        print(f"[STORAGE] Error loading predictions: {e}")
        return {'predictions': []}


def _save_predictions(data):
    """
    Save predictions to JSON file
    
    Args:
        data: Predictions data structure to save
    """
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[STORAGE] Error saving predictions: {e}")
        raise


def save_prediction(prediction_data):
    """
    Save a single prediction to storage
    
    Args:
        prediction_data: Dict containing prediction information
            - timestamp: ISO format timestamp
            - prediction: 0 (on-time) or 1 (late)
            - probability_late: Float probability of late delivery
            - product_category_id: Product category identifier
            - customer_email: Customer email (optional)
            - shipping_mode: Shipping mode used
            - days_real: Actual shipping days
            - days_scheduled: Scheduled shipping days
    
    Returns:
        str: Unique ID of saved prediction
    """
    try:
        # Load existing predictions
        data = _load_predictions()
        
        # Add unique ID
        prediction_data['id'] = str(uuid.uuid4())
        
        # Ensure timestamp is present
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now().isoformat()
        
        # Append new prediction
        data['predictions'].append(prediction_data)
        
        # Save back to file
        _save_predictions(data)
        
        print(f"[STORAGE] Saved prediction {prediction_data['id']}")
        return prediction_data['id']
    
    except Exception as e:
        print(f"[STORAGE] Error saving prediction: {e}")
        raise


def get_predictions_by_date(date_str=None):
    """
    Get all predictions for a specific date
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD' (default: today)
    
    Returns:
        list: List of predictions for the specified date
    """
    try:
        data = _load_predictions()
        
        # Default to today if no date provided
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Filter predictions by date
        filtered = []
        for pred in data['predictions']:
            pred_date = pred['timestamp'].split('T')[0]  # Extract date from ISO timestamp
            if pred_date == date_str:
                filtered.append(pred)
        
        print(f"[STORAGE] Found {len(filtered)} predictions for {date_str}")
        return filtered
    
    except Exception as e:
        print(f"[STORAGE] Error retrieving predictions by date: {e}")
        return []


def get_late_predictions_today():
    """
    Get all late delivery predictions for today
    
    Returns:
        list: List of late delivery predictions (prediction = 1)
    """
    try:
        today_predictions = get_predictions_by_date()
        
        # Filter for late deliveries only
        late_predictions = [p for p in today_predictions if p.get('prediction') == 1]
        
        print(f"[STORAGE] Found {len(late_predictions)} late predictions today")
        return late_predictions
    
    except Exception as e:
        print(f"[STORAGE] Error retrieving late predictions: {e}")
        return []


def get_category_statistics(date_str=None):
    """
    Get statistics grouped by product category for a specific date
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD' (default: today)
    
    Returns:
        dict: Statistics by product category
            {
                'category_id': {
                    'total': int,
                    'late': int,
                    'on_time': int,
                    'late_percentage': float
                }
            }
    """
    try:
        predictions = get_predictions_by_date(date_str)
        
        # Group by category
        stats = {}
        for pred in predictions:
            cat_id = pred.get('product_category_id')
            if cat_id is None:
                continue
            
            if cat_id not in stats:
                stats[cat_id] = {
                    'total': 0,
                    'late': 0,
                    'on_time': 0
                }
            
            stats[cat_id]['total'] += 1
            if pred.get('prediction') == 1:
                stats[cat_id]['late'] += 1
            else:
                stats[cat_id]['on_time'] += 1
        
        # Calculate percentages
        for cat_id in stats:
            total = stats[cat_id]['total']
            if total > 0:
                stats[cat_id]['late_percentage'] = round(
                    (stats[cat_id]['late'] / total) * 100, 2
                )
            else:
                stats[cat_id]['late_percentage'] = 0.0
        
        print(f"[STORAGE] Generated statistics for {len(stats)} categories")
        return stats
    
    except Exception as e:
        print(f"[STORAGE] Error generating category statistics: {e}")
        return {}


def clear_old_predictions(days=7):
    """
    Remove predictions older than specified days
    
    Args:
        days: Number of days to keep (default: 7)
    
    Returns:
        int: Number of predictions removed
    """
    try:
        data = _load_predictions()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter out old predictions
        original_count = len(data['predictions'])
        data['predictions'] = [
            p for p in data['predictions']
            if datetime.fromisoformat(p['timestamp']) > cutoff_date
        ]
        
        removed_count = original_count - len(data['predictions'])
        
        # Save cleaned data
        _save_predictions(data)
        
        print(f"[STORAGE] Removed {removed_count} predictions older than {days} days")
        return removed_count
    
    except Exception as e:
        print(f"[STORAGE] Error clearing old predictions: {e}")
        return 0


def get_storage_stats():
    """
    Get overall storage statistics
    
    Returns:
        dict: Storage statistics
    """
    try:
        data = _load_predictions()
        
        total = len(data['predictions'])
        late = sum(1 for p in data['predictions'] if p.get('prediction') == 1)
        on_time = total - late
        
        return {
            'total_predictions': total,
            'late_predictions': late,
            'on_time_predictions': on_time,
            'storage_file': str(STORAGE_FILE),
            'storage_exists': STORAGE_FILE.exists()
        }
    
    except Exception as e:
        print(f"[STORAGE] Error getting storage stats: {e}")
        return {}


# Testing function
if __name__ == "__main__":
    print("Testing Predictions Storage Module...")
    print("=" * 50)
    
    # Test 1: Save a prediction
    test_prediction = {
        'timestamp': datetime.now().isoformat(),
        'prediction': 1,
        'probability_late': 0.85,
        'product_category_id': 73,
        'customer_email': 'test@example.com',
        'shipping_mode': 'Standard Class',
        'days_real': 6,
        'days_scheduled': 4
    }
    
    pred_id = save_prediction(test_prediction)
    print(f"✅ Test 1 Passed: Saved prediction with ID: {pred_id}")
    
    # Test 2: Get today's predictions
    today_preds = get_predictions_by_date()
    print(f"✅ Test 2 Passed: Found {len(today_preds)} predictions today")
    
    # Test 3: Get late predictions
    late_preds = get_late_predictions_today()
    print(f"✅ Test 3 Passed: Found {len(late_preds)} late predictions")
    
    # Test 4: Get category statistics
    stats = get_category_statistics()
    print(f"✅ Test 4 Passed: Generated stats for {len(stats)} categories")
    
    # Test 5: Get storage stats
    storage_stats = get_storage_stats()
    print(f"✅ Test 5 Passed: Storage stats:")
    print(f"   Total predictions: {storage_stats['total_predictions']}")
    print(f"   Storage file: {storage_stats['storage_file']}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")