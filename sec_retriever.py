"""
SEC Filing Retriever for fetching and processing SEC filings
"""
import requests
import time
import logging
import re
from bs4 import BeautifulSoup
from utils import clean_text

# Set up logging
logger = logging.getLogger("sec_filing_qa")

def validate_cik(cik):
    """Clean and validate a CIK number"""
    cik_clean = ''.join(c for c in cik if c.isdigit())
    if not cik_clean or len(cik_clean) > 10:
        raise ValueError(f"Invalid CIK format: {cik}")
    return cik_clean

class SECFilingRetriever:
    def __init__(self, user_email="user@example.com", user_name="SEC Filing Analyzer"):
        self.headers = {
            "User-Agent": f"{user_name} ({user_email})",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        self.request_delay = 0.1
    
    def get_filings_list(self, cik, form_type="10-K", max_filings=10):
        """Get a list of available filings for a CIK"""
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            time.sleep(self.request_delay)
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                
                company_name = data.get("name", "Unknown Company")
                
                if "filings" not in data or "recent" not in data["filings"]:
                    return []
                
                filings_list = []
                for i, form in enumerate(data["filings"]["recent"]["form"]):
                    if form == form_type and len(filings_list) < max_filings:
                        accession_number = data["filings"]["recent"]["accessionNumber"][i]
                        filing_date = data["filings"]["recent"]["filingDate"][i]
                        primary_doc = data["filings"]["recent"]["primaryDocument"][i]
                        reporting_period = data["filings"]["recent"].get("reportDate", [None] * len(data["filings"]["recent"]["form"]))[i]
                        
                        accession_clean = accession_number.replace("-", "")
                        
                        document_url = f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
                        raw_filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
                        
                        filings_list.append({
                            "cik": cik,
                            "company_name": company_name,
                            "form_type": form_type,
                            "filing_date": filing_date,
                            "reporting_period": reporting_period or filing_date,
                            "document_url": document_url,
                            "raw_filing_url": raw_filing_url,
                            "accession_number": accession_number,
                            "display_name": f"{company_name} - {form_type} - {filing_date}"
                        })
                
                return filings_list
        except Exception as e:
            logger.error(f"Error retrieving filings list: {str(e)}")
            return []

    def extract_text_from_filing(self, filing_info):
        if not filing_info or "raw_filing_url" not in filing_info:
            return None
        
        try:
            url = filing_info["raw_filing_url"]
            
            response = requests.get(url, headers={
                "User-Agent": self.headers["User-Agent"],
                "Accept-Encoding": "gzip, deflate"
            }, timeout=60)
            time.sleep(self.request_delay)
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text(separator=' ', strip=True)
            
            cleaned_text = clean_text(text)
            
            logger.info(f"Extracted {len(cleaned_text)} characters of text")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None