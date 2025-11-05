"""
Flask API for Delivery Risk Prediction
Serves ML predictions as REST API endpoints
Stores predictions for later batch processing (DevOps Pipeline)
"""

from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.model_trainer import load_model, load_scaler, load_encoder, load_ordinal_encoder
from src.predictor import preprocess_new_data, predict
from src.predictions_store import save_prediction  # NEW: Storage function

app = Flask(__name__)

# Global variables for tracking metrics
prediction_count = 0
delay_count = 0

# Load model and artifacts at startup
print("[STARTUP] Loading trained model and artifacts...")
try:
    model = load_model()
    scaler = load_scaler()
    encoder = load_encoder()
    ord_encoder = load_ordinal_encoder()
    print("[STARTUP] ✅ All models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    model = scaler = encoder = ord_encoder = None


# ========== API ENDPOINTS ==========

@app.route('/')
def home():
    """Health check and API info"""
    return jsonify({
        'status': 'active',
        'message': 'Delivery Risk Prediction API v2.0 - DevOps Pipeline Ready',
        'features': [
            'AI-powered late delivery prediction',
            'Prediction storage for batch processing',
            'DevOps pipeline integration'
        ],
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'predict_batch': '/predict/batch (POST)',
            'metrics': '/metrics',
            'status': '/status'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'service': 'Delivery Risk Prediction'
    }), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Main prediction endpoint - ONLY predicts and stores
    Accepts JSON with delivery/order features
    Returns delivery risk prediction and saves to storage
    """
    global prediction_count, delay_count
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate that we have the required columns
        required_cols = [
            'Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
            'Benefit per order', 'Sales per customer', 'Category Id', 'Department Id',
            'Order Item Discount', 'Order Item Product Price', 'Order Item Profit Ratio',
            'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order',
            'Product Category Id', 'Product Price', 'shipping date (DateOrders)',
            'order date (DateOrders)', 'Shipping Mode'
        ]
        
        missing_cols = [col for col in required_cols if col not in data]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}',
                'required_columns': required_cols
            }), 400
        
        # Extract customer_email BEFORE preprocessing (not a model feature)
        customer_email = data.pop('customer_email', None)
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        print(f"\n[PREDICT] Received prediction request")
        print(f"[PREDICT] Input shape: {input_df.shape}")
        
        # Preprocess data
        print("[PREDICT] Preprocessing data...")
        X_new = preprocess_new_data(input_df, encoder, scaler, ord_encoder)
        
        # Make prediction
        print("[PREDICT] Making prediction...")
        predictions, probabilities = predict(model, X_new)
        
        prediction_value = predictions[0]
        prob_ontime = float(probabilities[0][0])
        prob_late = float(probabilities[0][1])
        
        # Update metrics
        prediction_count += 1
        if prediction_value == 1:
            delay_count += 1
            status = "Late Delivery Risk"
        else:
            status = "On-Time Delivery"
        
        # Prepare response
        result = {
            'status': status,
            'prediction': int(prediction_value),
            'probability_on_time': round(prob_ontime, 4),
            'probability_late_delivery': round(prob_late, 4),
            'confidence': round(max(prob_ontime, prob_late), 4),
            'timestamp': datetime.now().isoformat(),
            'model_version': '2.0'
        }
        
        print(f"[PREDICT] ✅ Prediction: {status}")
        print(f"[PREDICT] Probabilities - On-Time: {prob_ontime:.2%}, Late: {prob_late:.2%}")
        
        # ========== PHASE 1: STORE PREDICTION (NO EMAIL) ==========
        # Save prediction to storage for later batch processing
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction': int(prediction_value),
            'probability_late': round(prob_late, 4),
            'product_category_id': data.get('Product Category Id'),
            'customer_email': customer_email,
            'shipping_mode': data.get('Shipping Mode'),
            'days_real': data.get('Days for shipping (real)'),
            'days_scheduled': data.get('Days for shipment (scheduled)')
        }
        
        save_prediction(prediction_data)
        print(f"[STORAGE] ✅ Prediction saved to storage")
        
        result['stored'] = True
        result['storage_note'] = 'Prediction saved for batch processing'
        
        return jsonify(result), 200
    
    except ValueError as e:
        print(f"[ERROR] Value error: {e}")
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint - ONLY predicts and stores
    Accepts CSV data via file upload or JSON array
    Returns predictions for multiple records
    Saves all predictions to storage for later processing
    """
    global prediction_count, delay_count
    
    try:
        print("\n[BATCH PREDICT] Batch prediction request received")
        
        # Check if file uploaded or JSON provided
        if 'file' in request.files:
            # File upload mode
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            try:
                input_df = pd.read_csv(file, encoding='ISO-8859-1')
                print(f"[BATCH PREDICT] Loaded {len(input_df)} records from file")
            except Exception as e:
                return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
        
        elif request.is_json:
            # JSON array mode
            data = request.get_json()
            if not isinstance(data, list):
                return jsonify({'error': 'Expected JSON array'}), 400
            input_df = pd.DataFrame(data)
            print(f"[BATCH PREDICT] Loaded {len(input_df)} records from JSON")
        else:
            return jsonify({'error': 'Provide CSV file or JSON array'}), 400
        
        # Extract customer emails before preprocessing (if present)
        if 'customer_email' in input_df.columns:
            customer_emails = input_df.pop('customer_email')
        else:
            customer_emails = pd.Series([None] * len(input_df))
        
        # Preprocess all data
        print("[BATCH PREDICT] Preprocessing batch data...")
        X_new = preprocess_new_data(input_df, encoder, scaler, ord_encoder)
        
        # Make predictions
        print("[BATCH PREDICT] Making batch predictions...")
        predictions, probabilities = predict(model, X_new)
        
        # Update metrics
        num_predictions = len(predictions)
        num_late = int(np.sum(predictions))
        prediction_count += num_predictions
        delay_count += num_late
        
        # Prepare results and save to storage
        results = []
        for i in range(len(predictions)):
            result_item = {
                'record_id': i + 1,
                'prediction': int(predictions[i]),
                'status': 'Late Delivery Risk' if predictions[i] == 1 else 'On-Time Delivery',
                'probability_on_time': round(float(probabilities[i][0]), 4),
                'probability_late_delivery': round(float(probabilities[i][1]), 4),
                'confidence': round(float(max(probabilities[i])), 4)
            }
            results.append(result_item)
            
            # ========== PHASE 1: STORE EACH PREDICTION ==========
            # Save to storage for batch processing
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'prediction': int(predictions[i]),
                'probability_late': round(float(probabilities[i][1]), 4),
                'product_category_id': int(input_df.iloc[i].get('Product Category Id')),
                'customer_email': customer_emails.iloc[i] if i < len(customer_emails) else None,
                'shipping_mode': input_df.iloc[i].get('Shipping Mode'),
                'days_real': int(input_df.iloc[i].get('Days for shipping (real)')),
                'days_scheduled': int(input_df.iloc[i].get('Days for shipment (scheduled)'))
            }
            save_prediction(prediction_data)
        
        print(f"[STORAGE] ✅ {num_predictions} predictions saved to storage")
        
        response = {
            'total_predictions': int(num_predictions),
            'late_delivery_count': int(num_late),
            'on_time_count': int(num_predictions - num_late),
            'late_delivery_rate': round(num_late / num_predictions, 4) if num_predictions > 0 else 0,
            'predictions': results,
            'timestamp': datetime.now().isoformat(),
            'stored': True,
            'storage_note': 'All predictions saved for batch processing'
        }
        
        print(f"[BATCH PREDICT] ✅ Completed: {num_predictions} predictions")
        print(f"[BATCH PREDICT] Late Deliveries: {num_late}/{num_predictions}")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"[ERROR] Batch prediction error: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Get API metrics and statistics
    """
    return jsonify({
        'total_predictions': prediction_count,
        'late_delivery_predictions': delay_count,
        'on_time_predictions': prediction_count - delay_count,
        'late_delivery_rate': f"{(delay_count/prediction_count*100):.2f}%" if prediction_count > 0 else "0%",
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """
    Get API status and model information
    """
    return jsonify({
        'api_status': 'running',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'model_version': '2.0',
        'model_type': 'XGBoost Classifier',
        'model_accuracy': '97.4%',
        'features': {
            'ai_prediction': True,
            'prediction_storage': True,
            'batch_processing': True,
            'devops_pipeline_ready': True
        },
        'total_predictions_served': prediction_count,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/predictions/today', methods=['GET'])
def get_today_predictions_api():
    """
    Get all predictions for today (for GitHub Actions)
    """
    try:
        from src.predictions_store import get_predictions_by_date
        today = datetime.now().strftime('%Y-%m-%d')
        predictions = get_predictions_by_date(today)
        
        return jsonify({
            'date': today,
            'count': len(predictions),
            'predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/stats', methods=['GET'])
def get_predictions_stats_api():
    """
    Get category statistics for today (for GitHub Actions)
    """
    try:
        from src.predictions_store import get_predictions_by_date, get_category_statistics
        today = datetime.now().strftime('%Y-%m-%d')
        
        all_preds = get_predictions_by_date(today)
        stats = get_category_statistics(today)
        
        late_count = sum(1 for p in all_preds if p.get('prediction') == 1)
        
        return jsonify({
            'date': today,
            'total_predictions': len(all_preds),
            'late_count': late_count,
            'on_time_count': len(all_preds) - late_count,
            'category_statistics': stats
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run Flask app
    print("\n" + "="*70)
    print("🚀 DELIVERY RISK PREDICTION API v2.0 - DEVOPS PIPELINE")
    print("="*70)
    print(f"Server starting at: {datetime.now().isoformat()}")
    print("API Documentation: http://localhost:5000/")
    print("\nEndpoints:")
    print("  • GET  /                    - API Info")
    print("  • GET  /health              - Health Check")
    print("  • POST /predict             - Single Prediction + Storage")
    print("  • POST /predict/batch       - Batch Prediction + Storage")
    print("  • GET  /metrics             - Prediction Metrics")
    print("  • GET  /status              - API Status")
    print("\n✨ Features:")
    print("  • AI: 97.4% accurate delivery delay predictions")
    print("  • DevOps: Prediction storage for batch processing")
    print("  • Pipeline: Ready for GitHub Actions integration")
    print("="*70 + "\n")
    
    # Run with debug mode off for production
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)