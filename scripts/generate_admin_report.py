"""
Daily Admin Report Generator
Analyzes predictions and sends summary email to admin
Run daily via GitHub Actions at 11 PM
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predictions_store import (
    get_predictions_by_date,
    get_late_predictions_today,
    get_category_statistics
)
from src.notifier import send_admin_daily_report


def generate_report(date_str=None):
    """
    Generate and send daily admin report
    
    Args:
        date_str: Date string 'YYYY-MM-DD' (default: today)
    """
    print("\n" + "="*70)
    print("📊 GENERATING DAILY ADMIN REPORT")
    print("="*70)
    
    # Use today if no date provided
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Report Date: {date_str}")
    
    # Load predictions
    print("\n[1/5] Loading predictions...")
    all_predictions = get_predictions_by_date(date_str)
    late_predictions = [p for p in all_predictions if p.get('prediction') == 1]
    
    total = len(all_predictions)
    late_count = len(late_predictions)
    
    print(f"   Total Predictions: {total}")
    print(f"   Late Predictions: {late_count}")
    
    if total == 0:
        print("\n⚠️  No predictions found for today. Exiting.")
        return False
    
    # Get category statistics
    print("\n[2/5] Analyzing product categories...")
    category_stats = get_category_statistics(date_str)
    
    # Sort categories by late count (descending)
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: x[1]['late'],
        reverse=True
    )
    
    print(f"   Analyzed {len(sorted_categories)} categories")
    
    # Get top 3 problematic categories
    print("\n[3/5] Identifying top problematic categories...")
    top_3_categories = sorted_categories[:3]
    
    for i, (cat_id, stats) in enumerate(top_3_categories, 1):
        print(f"   {i}. Category {cat_id}: {stats['late']} late ({stats['late_percentage']:.1f}%)")
    
    # Calculate overall metrics
    print("\n[4/5] Calculating metrics...")
    late_rate = (late_count / total * 100) if total > 0 else 0
    print(f"   Overall Late Rate: {late_rate:.1f}%")
    
    # Prepare report data
    report_data = {
        'date': date_str,
        'total_predictions': total,
        'late_count': late_count,
        'on_time_count': total - late_count,
        'late_rate': late_rate,
        'top_categories': [
            {
                'category_id': cat_id,
                'late_count': stats['late'],
                'total_count': stats['total'],
                'late_percentage': stats['late_percentage']
            }
            for cat_id, stats in top_3_categories
        ]
    }
    
    # Send email to admin
    print("\n[5/5] Sending report to admin...")
    admin_email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
    
    success = send_admin_daily_report(admin_email, report_data)
    
    if success:
        print(f"   ✅ Report sent to {admin_email}")
        print("\n" + "="*70)
        print("✅ ADMIN REPORT GENERATION COMPLETE")
        print("="*70)
        return True
    else:
        print(f"   ❌ Failed to send report")
        print("\n" + "="*70)
        print("❌ ADMIN REPORT GENERATION FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Generate report for today
    success = generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)