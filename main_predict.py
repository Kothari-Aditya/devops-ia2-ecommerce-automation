"""
Main prediction pipeline orchestrator - ENHANCED VERSION
Supports both single record and batch CSV predictions.

Usage:
    # Batch prediction (CSV file)
    python main_predict.py data/raw/new_orders.csv
    
    # Single record prediction (interactive)
    python main_predict.py --single
    
    # Single record prediction (with values)
    python main_predict.py --single --type DEBIT --shipping-mode "Standard Class" ...
"""

import sys
import pandas as pd
import argparse
from src.model_trainer import load_model, load_scaler, load_encoder, load_ordinal_encoder
from src.predictor import preprocess_new_data, predict


def get_single_record_interactive():
    """
    Get single record input interactively from user.
    Prompts for only the ESSENTIAL features needed for prediction.
    """
    print("\n" + "="*70)
    print("SINGLE RECORD PREDICTION - INTERACTIVE MODE")
    print("="*70)
    print("Please provide the following information:")
    print("(Press Enter to use default values where applicable)")
    print("-"*70)
    
    # Essential features (minimum needed)
    record = {}
    
    # Payment Type
    print("\n1. Payment Type")
    print("   Options: DEBIT, CASH, TRANSFER, PAYMENT")
    record['Type'] = input("   Enter Type [default: DEBIT]: ").strip().upper() or "DEBIT"
    
    # Shipping days
    print("\n2. Shipping Information")
    record['Days for shipping (real)'] = float(input("   Days for shipping (real) [default: 3]: ").strip() or "3")
    record['Days for shipment (scheduled)'] = float(input("   Days for shipment (scheduled) [default: 4]: ").strip() or "4")
    
    # Shipping Mode
    print("\n3. Shipping Mode")
    print("   Options: Standard Class, Second Class, First Class, Same Day")
    record['Shipping Mode'] = input("   Enter Shipping Mode [default: Standard Class]: ").strip() or "Standard Class"
    
    # Financial info
    print("\n4. Financial Information")
    record['Benefit per order'] = float(input("   Benefit per order [default: 50.0]: ").strip() or "50.0")
    record['Sales per customer'] = float(input("   Sales per customer [default: 300.0]: ").strip() or "300.0")
    
    # Order details
    print("\n5. Order Details")
    record['Order Item Discount'] = float(input("   Order Item Discount [default: 0.0]: ").strip() or "0.0")
    record['Order Item Product Price'] = float(input("   Order Item Product Price [default: 100.0]: ").strip() or "100.0")
    record['Order Item Profit Ratio'] = float(input("   Order Item Profit Ratio [default: 0.2]: ").strip() or "0.2")
    record['Order Item Quantity'] = int(input("   Order Item Quantity [default: 1]: ").strip() or "1")
    record['Sales'] = float(input("   Sales [default: 100.0]: ").strip() or "100.0")
    record['Order Item Total'] = float(input("   Order Item Total [default: 100.0]: ").strip() or "100.0")
    record['Order Profit Per Order'] = float(input("   Order Profit Per Order [default: 20.0]: ").strip() or "20.0")
    
    # Product info
    print("\n6. Product Information")
    record['Product Price'] = float(input("   Product Price [default: 100.0]: ").strip() or "100.0")
    record['Category Id'] = int(input("   Category Id [default: 73]: ").strip() or "73")
    record['Department Id'] = int(input("   Department Id [default: 10]: ").strip() or "10")
    record['Product Category Id'] = int(input("   Product Category Id [default: 73]: ").strip() or "73")
    
    # Dates
    print("\n7. Date Information")
    record['shipping date (DateOrders)'] = input("   Shipping date (M/D/YYYY H:MM) [default: 2/3/2018 22:56]: ").strip() or "2/3/2018 22:56"
    record['order date (DateOrders)'] = input("   Order date (M/D/YYYY H:MM) [default: 2/1/2018 10:30]: ").strip() or "2/1/2018 10:30"
    
    return pd.DataFrame([record])


def create_sample_record():
    """
    Create a sample record with default values for quick testing.
    """
    return pd.DataFrame([{
        'Type': 'DEBIT',
        'Days for shipping (real)': 3,
        'Days for shipment (scheduled)': 4,
        'Benefit per order': 91.25,
        'Sales per customer': 314.64,
        'Category Id': 73,
        'Department Id': 10,
        'Order Item Discount': 0.0,
        'Order Item Product Price': 327.75,
        'Order Item Profit Ratio': 0.28,
        'Order Item Quantity': 1,
        'Sales': 327.75,
        'Order Item Total': 327.75,
        'Order Profit Per Order': 91.25,
        'Product Category Id': 73,
        'Product Price': 327.75,
        'shipping date (DateOrders)': '2/3/2018 22:56',
        'order date (DateOrders)': '2/1/2018 10:30',
        'Shipping Mode': 'Standard Class'
    }])


def display_input_data(df):
    """Display the input data in a readable format."""
    print("\nInput Data:")
    print("-" * 70)
    for col in df.columns:
        print(f"  {col}: {df[col].values[0]}")
    print("-" * 70)


def predict_single(record_df, model, encoder, scaler, ord_encoder):
    """
    Make prediction on a single record.
    """
    print("\n" + "="*70)
    print("PROCESSING SINGLE RECORD")
    print("="*70)
    
    # Preprocess
    X_new = preprocess_new_data(record_df, encoder, scaler, ord_encoder)
    
    # Predict
    predictions, probabilities = predict(model, X_new)
    
    # Display results
    result = predictions[0]
    prob_ontime = probabilities[0][0]
    prob_late = probabilities[0][1]
    
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"\nPrediction: {'🔴 LATE DELIVERY RISK' if result == 1 else '🟢 ON-TIME DELIVERY'}")
    print(f"\nProbabilities:")
    print(f"  • On-Time Delivery:  {prob_ontime:.1%} {'✓' if result == 0 else ''}")
    print(f"  • Late Delivery:     {prob_late:.1%} {'✓' if result == 1 else ''}")
    print("="*70)
    
    return result, prob_ontime, prob_late


def main():
    """
    End-to-end prediction pipeline supporting both single and batch modes.
    """
    print("="*70)
    print("DELIVERY RISK PREDICTION - PREDICTION PIPELINE")
    print("="*70)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict delivery risk')
    parser.add_argument('input_file', nargs='?', help='Path to CSV file for batch prediction')
    parser.add_argument('--single', action='store_true', help='Single record prediction (interactive)')
    parser.add_argument('--sample', action='store_true', help='Use sample record for quick testing')
    
    args = parser.parse_args()
    
    try:
        # ========== STEP 1: Load Trained Artifacts ==========
        print("\n[STEP 1] Loading trained model, scaler, and encoders...")
        model = load_model()
        scaler = load_scaler()
        encoder = load_encoder()
        ord_encoder = load_ordinal_encoder()
        
        # ========== STEP 2: Determine Mode ==========
        if args.sample:
            # Sample mode - quick test
            print("\n[STEP 2] Using sample record for quick testing...")
            new_data = create_sample_record()
            display_input_data(new_data)
            predict_single(new_data, model, encoder, scaler, ord_encoder)
            
        elif args.single:
            # Single record interactive mode
            new_data = get_single_record_interactive()
            predict_single(new_data, model, encoder, scaler, ord_encoder)
            
        elif args.input_file:
            # Batch CSV mode (original functionality)
            new_data_path = args.input_file
            
            print(f"\n[STEP 2] Loading new data from: {new_data_path}")
            new_data = pd.read_csv(new_data_path, encoding='ISO-8859-1')
            print(f"Loaded {len(new_data)} rows for prediction")
            
            # ========== STEP 3: Preprocess New Data ==========
            print("\n[STEP 3] Preprocessing new data...")
            X_new = preprocess_new_data(new_data, encoder, scaler, ord_encoder)
            
            # ========== STEP 4: Make Predictions ==========
            print("\n[STEP 4] Making predictions...")
            predictions, probabilities = predict(model, X_new)
            
            # ========== STEP 5: Prepare Results ==========
            print("\n[STEP 5] Preparing results...")
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Predicted_Late_Delivery_Risk': predictions,
                'Probability_On_Time': probabilities[:, 0],
                'Probability_Late_Delivery': probabilities[:, 1]
            })
            
            # Add prediction labels
            results['Prediction_Label'] = results['Predicted_Late_Delivery_Risk'].map({
                0: 'On-Time Delivery',
                1: 'Late Delivery Risk'
            })
            
            # Summary statistics
            late_count = (predictions == 1).sum()
            ontime_count = (predictions == 0).sum()
            late_percentage = (late_count / len(predictions)) * 100
            
            print("\n" + "="*70)
            print("PREDICTION RESULTS SUMMARY")
            print("="*70)
            print(f"Total orders predicted: {len(predictions)}")
            print(f"Predicted ON-TIME: {ontime_count} ({100-late_percentage:.1f}%)")
            print(f"Predicted LATE: {late_count} ({late_percentage:.1f}%)")
            print("="*70)
            
            # Display sample predictions
            print("\nSample predictions (first 10 rows):")
            print(results.head(10).to_string(index=False))
            
            # ========== STEP 6: Save Results ==========
            output_path = new_data_path.replace('.csv', '_predictions.csv')
            
            # Combine with original data (optional - include key columns)
            if 'Order Id' in new_data.columns:
                results['Order_Id'] = new_data['Order Id'].values
            
            results.to_csv(output_path, index=False)
            print(f"\n✅ Predictions saved to: {output_path}")
        
        else:
            # No arguments provided
            print("\n❌ ERROR: No input provided")
            print("\nUsage options:")
            print("  1. Batch CSV prediction:")
            print("     python main_predict.py data/raw/new_orders.csv")
            print("\n  2. Single record (interactive):")
            print("     python main_predict.py --single")
            print("\n  3. Quick test (sample record):")
            print("     python main_predict.py --sample")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("✅ PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("\nMake sure:")
        print("  1. The input CSV file exists")
        print("  2. You have trained the model (run main_train.py first)")
        sys.exit(1)
    
    except KeyError as e:
        print(f"\n❌ ERROR: Missing column in data - {e}")
        print("\nMake sure the input data has the same columns as training data")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()