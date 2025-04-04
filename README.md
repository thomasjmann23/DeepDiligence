# SEC Filing Q&A Application

This application allows users to query SEC filings using LangChain and Google's Gemini AI. It provides a user-friendly interface for retrieving, analyzing, and asking questions about SEC filings.

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
   streamlit run main.py
   ```

## Usage

1. Configure your settings and API key (a default Gemini API key is provided)
2. Select one or more companies and filing types
3. Choose which filings to analyze
4. Ask questions about the filings

## Project Structure

- `main.py` - Streamlit web application and UI components
- `sec_retriever.py` - SEC API interactions for retrieving filing data
- `qa_system.py` - LangChain and Gemini integration for Q&A functionality
- `utils.py` - Utility functions used across the application

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.