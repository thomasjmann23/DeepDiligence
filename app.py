"""
DeepDiligence SEC Filing Q&A App - Using LangChain and Gemini
Features:
- Default API key
- Compare multiple reports from the same company
- Compare reports from multiple companies
"""
import os
import streamlit as st
import requests
import time
import logging
from typing import Dict, Any, List, Optional
import re
from bs4 import BeautifulSoup
import pandas as pd

# LangChain and Gemini imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sec_filing_qa")

# Constants
KNOWN_CIKS = {
    "320193": "Apple Inc",
    "1652044": "Alphabet Inc (Google)",
    "1326801": "Meta Platforms Inc (Facebook)",
    "1018724": "Amazon.com Inc",
    "789019": "Microsoft Corp",
    "1318605": "Tesla Inc"
}

# Default API key (your requested key)
DEFAULT_GEMINI_API_KEY = "AIzaSyA56W2eGvyaHQeSf4Lc3o7oY6hPCy2EG38"

# Helper Functions
def validate_cik(cik):
    """Clean and validate a CIK number"""
    cik_clean = ''.join(c for c in cik if c.isdigit())
    if not cik_clean or len(cik_clean) > 10:
        raise ValueError(f"Invalid CIK format: {cik}")
    return cik_clean

def clean_text(text):
    """Clean text from HTML content"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n\n', text)
    return text.strip()

def format_sources(sources, max_length=300):
    """Format source documents for display"""
    formatted = []
    for source in sources:
        if len(source) > max_length:
            formatted.append(f"{source[:max_length]}...")
        else:
            formatted.append(source)
    return formatted

# SEC Filing Retriever
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

# SEC Filing QA System
class SECFilingQA:
    def __init__(self, gemini_api_key):
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.llm = ChatGoogleGenerativeAI(
            model='models/gemini-1.5-flash',
            temperature=0.2
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # Standard filing QA prompt template
        self.filing_prompt_template = """
        You are a financial analyst assistant specialized in SEC filings. Use the following pieces of context to answer the question.
        If the answer isn't in the context, say "I could not find this information in the filing."
        
        Context:
        {context}
        
        Question:
        {question}

        If the question relates to predicting stocks, future speculation, fraud, or any illegal activity, say "This request is outside the scope of the intended use of this application."
        """
        
        # Multi-filing comparison prompt template
        self.multi_filing_prompt_template = """
        You are a financial analyst assistant specialized in SEC filings. Use the following pieces of context from multiple SEC filings to answer the question at the end.
        
        The context contains information from these filings:
        {filing_descriptions}
        
        When referring to specific filings in your answer, please cite them clearly.
        If the answer isn't in the contexts, say "I could not find this information in the filings."
        
        Contexts:
        {context}
        
        Question:
        {question}
        
        If the question relates to predicting stocks, future speculation, fraud, or any illegal activity, say "This request is outside the scope of the intended use of this application."
        """
        
        self.qa_chain = None
        self.company_info = None
        self.retriever = None
        self.mode = "single" # Either "single" or "multi"
        self.filings_info = []
    
    def process_filing_text(self, text, company_info):
        """Process a single filing"""
        self.mode = "single"
        self.company_info = company_info
        self.filings_info = [company_info]
        
        # Chunk the text
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Create FAISS vector store with Gemini embeddings
        vectorstore = FAISS.from_texts(chunks, self.embedding_model)
        
        # Create retriever and store it for later use
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create QA chain
        prompt = PromptTemplate(
            template=self.filing_prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        
        return True
    
    def process_multiple_filings(self, texts_and_infos):
        """Process multiple filings for comparison
        Args:
            texts_and_infos: List of tuples (text, filing_info)
        """
        self.mode = "multi"
        self.filings_info = [info for _, info in texts_and_infos]
        
        # For each filing, add a prefix to each chunk to identify the source
        all_chunks = []
        for text, info in texts_and_infos:
            chunks = self.text_splitter.split_text(text)
            # Add filing identifier to each chunk
            prefixed_chunks = [
                f"FILING: {info['company_name']} - {info['form_type']} - {info['filing_date']}\n\n{chunk}"
                for chunk in chunks
            ]
            all_chunks.extend(prefixed_chunks)
            logger.info(f"Added {len(chunks)} chunks from {info['company_name']} {info['form_type']} ({info['filing_date']})")
        
        # Create FAISS vector store with all chunks
        vectorstore = FAISS.from_texts(all_chunks, self.embedding_model)
        
        # More chunks for multiple filings
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        
        # Create filing descriptions for prompt
        filing_descriptions = "\n".join([
            f"- {info['company_name']} {info['form_type']} ({info['filing_date']})"
            for info in self.filings_info
        ])
        
        # Create multi-filing QA chain with dynamic filing descriptions
        prompt = PromptTemplate(
            template=self.multi_filing_prompt_template,
            input_variables=["context", "question", "filing_descriptions"]
        )
        
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        
        return True
    
    def ask_question(self, question):
        if not self.qa_chain or not self.retriever:
            return {"answer": "Error: No filings have been processed yet.", "sources": []}
        
        try:
            # Get relevant documents from the retriever
            docs = self.retriever.get_relevant_documents(question)
            
            if self.mode == "single":
                # Run the chain with docs and question
                result = self.qa_chain({"input_documents": docs, "question": question})
                
                # Extract answer
                answer = result.get("output_text", "No answer found")
                
                # Enrich answer with filing information
                enriched_answer = f"Based on the {self.company_info.get('form_type', 'filing')} " + \
                                f"for {self.company_info.get('company_name', 'the company')} " + \
                                f"dated {self.company_info.get('filing_date', '')}:\n\n{answer}"
            else:
                # For multi-filing mode, include filing descriptions
                filing_descriptions = "\n".join([
                    f"- {info['company_name']} {info['form_type']} ({info['filing_date']})"
                    for info in self.filings_info
                ])
                
                # Run the chain with docs, question, and filing descriptions
                result = self.qa_chain({
                    "input_documents": docs, 
                    "question": question,
                    "filing_descriptions": filing_descriptions
                })
                
                # Extract answer
                answer = result.get("output_text", "No answer found")
                answer = re.sub(r'\$', r'\\$', answer)  # Adds a backslash before every dollar sign
                enriched_answer = f"Based on analysis of {len(self.filings_info)} SEC filings:\n\n{answer}"
            
            # Use the retrieved docs as sources
            sources = [doc.page_content for doc in docs]
            
            return {"answer": enriched_answer, "sources": sources}
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {"answer": f"Error: {str(e)}", "sources": []}

# Streamlit App
def main():
    st.set_page_config(
        page_title="DeepDiligence SEC Filing Q&A",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if 'filing_processed' not in st.session_state:
        st.session_state.filing_processed = False
    
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = DEFAULT_GEMINI_API_KEY
    
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    
    if 'company_info' not in st.session_state:
        st.session_state.company_info = None
    
    if 'available_filings' not in st.session_state:
        st.session_state.available_filings = {}
        
    if 'selected_filings' not in st.session_state:
        st.session_state.selected_filings = []

    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = "single"
        
    # App title and description
    st.title("DeepDiligence SEC Filing Analysis")
    st.markdown("""
    This app allows you to ask questions about SEC filings using LangChain and Google's Gemini AI.
    
    **Features:**
    - Query a single SEC filing
    - Compare multiple filings from the same company
    - Compare filings across multiple companies
    
    **How to use:**
    1. Configure your settings and API key
    2. Select one or more companies and filing types
    3. Choose which filings to analyze
    4. Ask questions about the filings
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Gemini API key input with default
        api_key = st.text_input("Gemini API Key", 
                               type="password",
                               value=st.session_state.gemini_api_key,
                               help="Get your API key from Google AI Studio")
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        # Comparison mode selection
        comparison_mode = st.radio(
            "Analysis Mode",
            options=["Single Filing", "Multiple Filings from One Company", "Multiple Companies Comparison"],
            index=0,
            help="Select how you want to analyze filings"
        )
        
        # Set comparison mode
        if comparison_mode == "Single Filing":
            st.session_state.comparison_mode = "single"
        elif comparison_mode == "Multiple Filings from One Company":
            st.session_state.comparison_mode = "multi_same"
        else:
            st.session_state.comparison_mode = "multi_different"
            
        # Company selection section
        st.subheader("Company Selection")
        
        # Input for CIK numbers
        if st.session_state.comparison_mode == "multi_different":
            # Multiple companies - multiselect from known CIKs
            company_options = {f"{name}": cik for cik, name in KNOWN_CIKS.items()}
            selected_companies = st.multiselect(
                "Select Companies",
                options=list(company_options.keys()),
                help="Select companies to compare"
            )
            selected_ciks = [company_options[company] for company in selected_companies]
        else:
            # Single company selection
            cik_input_method = st.radio(
                "CIK Input Method",
                options=["Manual Entry", "Select from Known CIKs"],
                index=1
            )
            
            if cik_input_method == "Manual Entry":
                cik_input = st.text_input(
                    "Enter CIK number",
                    help="Enter CIK number without leading zeros."
                )
                selected_ciks = [cik_input] if cik_input else []
            else:
                # Create a selectbox with known CIKs
                cik_options = {f"{name}": cik for cik, name in KNOWN_CIKS.items()}
                selected_company = st.selectbox(
                    "Select Company",
                    options=list(cik_options.keys()),
                    help="Select a company from the list."
                )
                selected_ciks = [cik_options[selected_company]] if selected_company else []
        
        # Form type selection
        form_type = st.selectbox(
            "Filing Type",
            options=["10-K", "10-Q", "8-K"],
            index=0,
            help="Select the type of filing to analyze"
        )
        
        # User email for SEC.gov
        user_email = st.text_input(
            "Email for SEC.gov",
            value="user@example.com",
            help="Required for SEC.gov API access"
        )
        
        # Search filings button
        search_button = st.button("Search Available Filings", type="primary")
    
    # Search for available filings when button is clicked
    if search_button and selected_ciks and st.session_state.gemini_api_key:
        try:
            with st.spinner("Searching for available filings..."):
                # Initialize retriever
                retriever = SECFilingRetriever(user_email=user_email)
                
                # Clear previous selections
                st.session_state.selected_filings = []
                st.session_state.available_filings = {}
                
                # Get filings for each selected CIK
                for cik in selected_ciks:
                    cik = validate_cik(cik)
                    filings = retriever.get_filings_list(cik, form_type=form_type, max_filings=10)
                    
                    if filings:
                        # Store available filings in session state
                        st.session_state.available_filings[cik] = filings
                    else:
                        st.warning(f"No {form_type} filings found for CIK {cik}")
                
                if st.session_state.available_filings:
                    st.success(f"Found filings for {len(st.session_state.available_filings)} companies")
                
        except Exception as e:
            st.error(f"Error searching filings: {str(e)}")
    
    # Initialize process_button as False by default to avoid UnboundLocalError
    process_button = False
    
    # Display available filings for selection
    if st.session_state.available_filings:
        st.subheader("Available Filings")
        
        filing_selection_container = st.container()
        
        with filing_selection_container:
            # Create selection widgets based on comparison mode
            if st.session_state.comparison_mode == "single":
                # Single filing mode - just pick one
                all_filings = []
                for cik, filings in st.session_state.available_filings.items():
                    all_filings.extend(filings)
                
                filing_options = {filing["display_name"]: filing for filing in all_filings}
                selected_filing_name = st.selectbox(
                    "Select Filing to Analyze",
                    options=list(filing_options.keys()),
                    help="Select a filing to analyze"
                )
                
                if selected_filing_name:
                    st.session_state.selected_filings = [filing_options[selected_filing_name]]
            
            elif st.session_state.comparison_mode == "multi_same":
                # Multiple filings from same company
                if len(st.session_state.available_filings) == 1:
                    cik = list(st.session_state.available_filings.keys())[0]
                    filings = st.session_state.available_filings[cik]
                    
                    filing_options = {filing["display_name"]: filing for filing in filings}
                    selected_filing_names = st.multiselect(
                        f"Select {form_type} Filings to Compare",
                        options=list(filing_options.keys()),
                        help=f"Select multiple {form_type} filings to compare"
                    )
                    
                    st.session_state.selected_filings = [filing_options[name] for name in selected_filing_names]
                else:
                    st.warning("Please select only one company for this comparison mode")
            
            else:  # multi_different
                # Multiple companies comparison - select one filing per company
                st.write("Select one filing from each company:")
                
                selected_filings = []
                for cik, filings in st.session_state.available_filings.items():
                    company_name = filings[0]["company_name"] if filings else f"CIK: {cik}"
                    
                    filing_options = {filing["display_name"]: filing for filing in filings}
                    selected_filing_name = st.selectbox(
                        f"Select filing for {company_name}",
                        options=list(filing_options.keys()),
                        key=f"company_{cik}"
                    )
                    
                    if selected_filing_name:
                        selected_filings.append(filing_options[selected_filing_name])
                
                st.session_state.selected_filings = selected_filings
            
            # Process filings button
            selected_count = len(st.session_state.selected_filings)
            process_button = st.button(
                f"Process {selected_count} Selected Filing{'s' if selected_count != 1 else ''}",
                disabled=selected_count == 0
            )
    
    # Process filings when button is clicked
    if process_button and st.session_state.selected_filings and st.session_state.gemini_api_key:
        try:
            with st.spinner(f"Processing {len(st.session_state.selected_filings)} filings..."):
                # Initialize retriever and QA system
                retriever = SECFilingRetriever(user_email=user_email)
                qa_system = SECFilingQA(st.session_state.gemini_api_key)
                
                if len(st.session_state.selected_filings) == 1:
                    # Single filing mode
                    filing_info = st.session_state.selected_filings[0]
                    
                    # Extract text from filing
                    filing_text = retriever.extract_text_from_filing(filing_info)
                    
                    if filing_text:
                        # Process filing text
                        qa_system.process_filing_text(filing_text, filing_info)
                        
                        # Store QA system in session state
                        st.session_state.qa_system = qa_system
                        st.session_state.filing_processed = True
                        st.session_state.company_info = filing_info
                        
                        st.success(f"Successfully processed {filing_info['form_type']} filing for {filing_info['company_name']} (dated {filing_info['filing_date']})")
                    else:
                        st.error("Failed to extract text from filing")
                else:
                    # Multiple filings mode
                    texts_and_infos = []
                    
                    # Extract text from each filing
                    progress_bar = st.progress(0)
                    for i, filing_info in enumerate(st.session_state.selected_filings):
                        # Update progress
                        progress = (i / len(st.session_state.selected_filings))
                        progress_bar.progress(progress, f"Processing {filing_info['company_name']} {filing_info['form_type']}")
                        
                        # Extract text
                        filing_text = retriever.extract_text_from_filing(filing_info)
                        
                        if filing_text:
                            texts_and_infos.append((filing_text, filing_info))
                        else:
                            st.warning(f"Failed to extract text from {filing_info['company_name']} {filing_info['form_type']} ({filing_info['filing_date']})")
                    
                    progress_bar.progress(1.0, "Processing complete")
                    
                    if texts_and_infos:
                        # Process multiple filings
                        qa_system.process_multiple_filings(texts_and_infos)
                        
                        # Store QA system in session state
                        st.session_state.qa_system = qa_system
                        st.session_state.filing_processed = True
                        
                        company_names = [info["company_name"] for _, info in texts_and_infos]
                        st.success(f"Successfully processed {len(texts_and_infos)} filings from {', '.join(set(company_names))}")
                    else:
                        st.error("Failed to extract text from any filings")
                
        except Exception as e:
            st.error(f"Error processing filings: {str(e)}")
    
    # Display filing information and Q&A interface
    if st.session_state.filing_processed:
        st.subheader("Filing Information")
        
        # Display filing information based on mode
        if len(st.session_state.selected_filings) == 1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Company", st.session_state.company_info["company_name"])
            with col2:
                st.metric("Filing Type", st.session_state.company_info["form_type"])
            with col3:
                st.metric("Filing Date", st.session_state.company_info["filing_date"])
            
            st.markdown(f"[View Filing on SEC.gov]({st.session_state.company_info['document_url']})")
        else:
            # For multiple filings, show a table
            filing_data = []
            for filing in st.session_state.selected_filings:
                filing_data.append({
                    "Company": filing["company_name"],
                    "Filing Type": filing["form_type"],
                    "Filing Date": filing["filing_date"]
                })
            
            df = pd.DataFrame(filing_data)
            
            # Display the dataframe
            st.dataframe(df, use_container_width=True)
            
            # Add links to the filings
            st.subheader("Filing Links")
            for i, filing in enumerate(st.session_state.selected_filings):
                st.markdown(f"[{filing['company_name']} - {filing['form_type']} ({filing['filing_date']})]({filing['document_url']})")
        
        # Question input
        st.subheader("Ask a Question")
        
        # Provide prompt suggestions based on comparison mode
        if len(st.session_state.selected_filings) > 1:
            if st.session_state.comparison_mode == "multi_same":
                st.info("Example questions: 'How has revenue changed across these quarters?' or 'Compare the risk factors across these filings.'")
            else:  # multi_different
                st.info("Example questions: 'Compare cash positions across these companies' or 'What do all these companies say about inflation?'")
        
        question = st.text_input("Enter your question about the filing(s)")
        
        if question:
            with st.spinner("Generating answer..."):
                # Ask question
                result = st.session_state.qa_system.ask_question(question)
                
                # Display answer
                st.markdown("### Answer")
                st.write(result["answer"])
                
                # Display sources
                with st.expander("View Sources", expanded=False):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"```\n{source[:300]}...\n```")
    
    # Show instructions if no filing processed
    if not st.session_state.filing_processed and not st.session_state.available_filings:
        st.info("Please select companies, choose a filing type, and click 'Search Available Filings' to begin")
    
    # Footer
    st.markdown("---")
    st.caption("DeepDiligence - SEC Filing Analysis App - Using LangChain and Gemini")

if __name__ == "__main__":
    main()
