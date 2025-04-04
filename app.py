"""
Enhanced SEC Filing Q&A App - Using LangChain and Gemini
Features:
- Default API key
- Compare multiple reports from the same company
- Compare reports from multiple companies
"""
import streamlit as st
import pandas as pd
import logging
from sec_retriever import SECFilingRetriever, validate_cik
from qa_system import SECFilingQA

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

# Default API key
DEFAULT_GEMINI_API_KEY = "AIzaSyA56W2eGvyaHQeSf4Lc3o7oY6hPCy2EG38"

# Streamlit App
def main():
    st.set_page_config(
        page_title="Enhanced SEC Filing Q&A",
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
    st.title("Enhanced SEC Filing Question & Answer")
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
    st.caption("Enhanced SEC Filing Q&A App - Using LangChain and Gemini")

if __name__ == "__main__":
    main()
