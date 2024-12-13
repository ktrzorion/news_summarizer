import asyncio
import aiohttp
import os
import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('async_company_reports.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Local API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# Email Configuration from environment variables
EMAIL_CONFIG = {
    'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
    'sender_email': os.getenv('EMAIL_SENDER'),
    'sender_password': os.getenv('EMAIL_PASSWORD'),
    'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(',')
}

async def analyze_company(session: aiohttp.ClientSession, company_name: str, hours_ago: int = 24, max_articles: int = 5) -> Dict:
    """
    Asynchronously generate a report for a single company
    
    :param session: Aiohttp client session
    :param company_name: Name of the company to analyze
    :param hours_ago: Number of hours to look back for news
    :param max_articles: Maximum number of articles to process
    :return: Company report or error information
    """
    try:
        # Prepare payload
        payload = {
            "company_name": company_name,
            "hours_ago": hours_ago,
            "max_articles": max_articles
        }
        
        # Make async POST request
        async with session.post(f"{API_BASE_URL}/analyze-company/", json=payload) as response:
            # Check for successful response
            if response.status == 200:
                report = await response.json()
                report['generated_at'] = datetime.now().isoformat()
                report['status'] = 'success'
                logger.info(f"Successfully generated report for {company_name}")
                return report
            else:
                # Handle error responses
                error_text = await response.text()
                logger.error(f"Failed to generate report for {company_name}: {response.status} - {error_text}")
                return {
                    'company_name': company_name,
                    'status': 'error',
                    'status_code': response.status,
                    'error_message': error_text,
                    'generated_at': datetime.now().isoformat()
                }
    
    except aiohttp.ClientError as e:
        logger.error(f"Network error for {company_name}: {e}")
        return {
            'company_name': company_name,
            'status': 'error',
            'error_message': str(e),
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Unexpected error for {company_name}: {e}")
        return {
            'company_name': company_name,
            'status': 'error',
            'error_message': str(e),
            'generated_at': datetime.now().isoformat()
        }

async def generate_company_reports(clients: List[str], hours_ago: int = 24, max_articles: int = 5):
    """
    Generate reports for multiple companies using async processing
    
    :param clients: List of company names
    :param hours_ago: Number of hours to look back for news
    :param max_articles: Maximum number of articles to process
    :return: List of company reports
    """
    # Create a single client session for all requests
    async with aiohttp.ClientSession() as session:
        # Create tasks for each company
        tasks = [
            analyze_company(session, company, hours_ago, max_articles) 
            for company in clients
        ]
        
        # Run all tasks concurrently
        reports = await asyncio.gather(*tasks)
        
        return reports

def save_reports_to_file(reports: List[Dict], filename: str = None):
    """
    Save reports to a JSON file
    
    :param reports: List of company reports
    :param filename: Optional filename, defaults to date-based filename
    :return: Path to the saved file
    """
    if not filename:
        # Create a filename using current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"company_reports_{current_date}.json"
    
    try:
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        filepath = os.path.join('reports', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2)
        
        logger.info(f"Reports saved to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Failed to save reports: {e}")
        return None

def send_email_with_report(filepath: str):
    """
    Send email with the generated report as an attachment
    
    :param filepath: Path to the report file
    :return: Boolean indicating success or failure
    """
    # Validate email configuration
    if not all([
        EMAIL_CONFIG['sender_email'], 
        EMAIL_CONFIG['sender_password'], 
        EMAIL_CONFIG['recipients']
    ]):
        logger.error("Email configuration is incomplete. Check your .env file.")
        return False

    try:
        # Create multipart message
        message = MIMEMultipart()
        message['From'] = EMAIL_CONFIG['sender_email']
        message['To'] = ', '.join(EMAIL_CONFIG['recipients'])
        message['Subject'] = f"Company Reports - {datetime.now().strftime('%Y-%m-%d')}"

        # Email body
        body = """
        Please find attached the company reports generated today.
        
        Generated by Async Company Reports Script
        """
        message.attach(MIMEText(body, 'plain'))

        # Attach the report file
        with open(filepath, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition', 
            f"attachment; filename= {os.path.basename(filepath)}"
        )
        message.attach(part)

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()  # Enable security
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.sendmail(
                EMAIL_CONFIG['sender_email'], 
                EMAIL_CONFIG['recipients'], 
                message.as_string()
            )
        
        logger.info(f"Email sent successfully with report: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

def main():
    # List of smaller tech companies to generate reports for
    clients = [
        "HCL Technologies", 
        "Infosys", 
        "Wipro", 
        "Cognizant", 
        "Tech Mahindra", 
        "Persistent Systems", 
        "Mphasis",
        "Mindtree",
        "Zensar Technologies",
        "NIIT Technologies"
    ]
    
    # Run the async report generation
    reports = asyncio.run(generate_company_reports(clients))
    
    # Save reports to file
    saved_filename = save_reports_to_file(reports)
    
    # Print summary
    print("\nReport Generation Summary:")
    for report in reports:
        status = report.get('status', 'Unknown')
        company = report.get('company_name', 'Unknown Company')
        if status == 'error':
            print(f"{company}: {status} - {report.get('error_message', 'Unknown error')}")
        else:
            print(f"{company}: {status}")
    
    if saved_filename:
        print(f"\nReports saved to: {saved_filename}")
        
        # Send email with the report
        email_result = send_email_with_report(saved_filename)
        if email_result:
            print("\nReport email sent successfully!")
        else:
            print("\nFailed to send report email.")

if __name__ == "__main__":
    main()