# DeepDiligence: SEC Filing Q&A Application

This application allows users to query SEC filings using LangChain and Google's Gemini AI. It provides a user-friendly interface for retrieving, analyzing, and asking questions about 1 or more SEC filings.

## Features

- Query a single SEC filing
- Compare multiple filings from the same company
- Compare filings across multiple companies
- Natural language Q&A using Gemini AI

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Configure your settings and API key (a default Gemini API key is provided)
2. Select one or more companies and filing types
3. Choose which filings to analyze
4. Ask questions about the filings

## Project Structure

- `app.py` - Streamlit web application, SEC EDGAR API, LangChain and FAISS, Gemini

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Google Generative AI Python SDK
- BeautifulSoup4
- FAISS-CPU
- Other dependencies listed in requirements.txt

## Limitations

- The application uses the SEC.gov API, which has rate limits
- Complex questions may require multiple queries
- The default API key has usage limitations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
