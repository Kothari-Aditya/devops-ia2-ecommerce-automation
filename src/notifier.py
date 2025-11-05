"""
Email Notification Module using Mailtrap
Sends customer alerts and admin reports
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mailtrap Configuration
MAILTRAP_HOST = os.getenv('MAILTRAP_HOST', 'sandbox.smtp.mailtrap.io')
MAILTRAP_PORT = int(os.getenv('MAILTRAP_PORT', 2525))
MAILTRAP_USERNAME = os.getenv('MAILTRAP_USERNAME', '')
MAILTRAP_PASSWORD = os.getenv('MAILTRAP_PASSWORD', '')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'noreply@deliveryalert.com')
SENDER_NAME = os.getenv('SENDER_NAME', 'E-Commerce Delivery Alert')


def send_admin_daily_report(admin_email, report_data):
    """
    Send daily admin report with product category analysis
    
    Args:
        admin_email: Admin email address
        report_data: Dict containing report information
    
    Returns:
        bool: True if email sent successfully
    """
    try:
        date = report_data['date']
        total = report_data['total_predictions']
        late = report_data['late_count']
        on_time = report_data['on_time_count']
        late_rate = report_data['late_rate']
        top_categories = report_data['top_categories']
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'📊 Daily Delivery Risk Report - {date}'
        msg['From'] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg['To'] = admin_email
        
        # Build top categories HTML
        categories_html = ""
        for i, cat in enumerate(top_categories, 1):
            cat_id = cat['category_id']
            cat_late = cat['late_count']
            cat_total = cat['total_count']
            cat_pct = cat['late_percentage']
            
            categories_html += f"""
                <li style="margin-bottom: 10px;">
                    <strong>Category {cat_id}:</strong> {cat_late} late out of {cat_total} 
                    <span style="color: #d9534f; font-weight: bold;">({cat_pct:.1f}%)</span>
                </li>
            """
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 700px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #2c3e50;">📊 Daily Delivery Risk Report</h2>
                <p style="font-size: 14px; color: #666;">Report Date: <strong>{date}</strong></p>
                
                <div style="background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3>📈 Daily Summary</h3>
                    <ul>
                        <li><strong>Total Predictions:</strong> {total}</li>
                        <li><strong>Late Deliveries:</strong> <span style="color: #e74c3c;">{late} ({late_rate:.1f}%)</span></li>
                        <li><strong>On-Time Deliveries:</strong> <span style="color: #27ae60;">{on_time}</span></li>
                    </ul>
                </div>
                
                <div style="background-color: #fee; padding: 20px; border-left: 4px solid #e74c3c; margin: 20px 0;">
                    <h3>🔴 Top 3 Problematic Categories</h3>
                    <ul style="list-style: none; padding: 0;">
                        {categories_html}
                    </ul>
                </div>
                
                <div style="background-color: #e8f4f8; padding: 20px; margin: 20px 0;">
                    <h3>💡 Recommendations</h3>
                    <ul>
                        <li>Review logistics for high-risk categories</li>
                        <li>Consider upgrading shipping for problematic categories</li>
                        <li>Implement proactive customer communication</li>
                    </ul>
                </div>
                
                <hr style="margin: 30px 0;">
                <p style="font-size: 12px; color: #999; text-align: center;">
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP(MAILTRAP_HOST, MAILTRAP_PORT) as server:
            server.starttls()
            server.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
            server.sendmail(SENDER_EMAIL, admin_email, msg.as_string())
        
        print(f"[NOTIFICATION] ✅ Daily admin report sent to {admin_email}")
        return True
    
    except Exception as e:
        print(f"[NOTIFICATION] ❌ Failed to send admin report: {e}")
        return False