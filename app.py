import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API KEY
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # Use correct key format
llm = ChatGroq(model_name="llama3-8b-8192")  # Correct spelling

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="mistral")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# User input field
user_prompt = st.text_input("Enter Your Query from research paper")  # Corrected text input usage

# Button to create document embeddings
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

# Ensure vector embeddings are created before querying
if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})  # Use 'input' as defined in prompt
        st.write(f"Response Time: {time.process_time() - start} seconds")

        # Display the answer
        st.write(response['answer'])

        # Display document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please create the vector database first by clicking 'Document Embedding'.")
