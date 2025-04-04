"""
SEC Filing Q&A System using LangChain and Gemini
"""
import os
import re
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Set up logging
logger = logging.getLogger("sec_filing_qa")

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