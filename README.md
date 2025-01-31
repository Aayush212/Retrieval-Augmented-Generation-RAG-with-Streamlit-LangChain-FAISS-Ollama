# Retrieval-Augmented-Generation-RAG-with-Streamlit-LangChain-FAISS-Ollama

This repository contains a Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, and Ollama embeddings. The app enables users to query research papers, leveraging a vector database for semantic search and generating responses using a LLM (Llama 3 via Groq API).

## Features

1. Load and process PDFs using PyPDFDirectoryLoader

2. Chunk and embed text using RecursiveCharacterTextSplitter and OllamaEmbeddings

3. Store embeddings in a FAISS vector database

4. Retrieve relevant documents using semantic search

5. Generate responses using ChatGroq (Llama 3 model)

6. Streamlit UI for user interaction

## Installation

### Prerequisites

  Python 3.9+

  Install Ollama and pull embedding model
```
curl -fsSL https://ollama.com/install.sh | sh  # Install Ollama (Linux/macOS)
ollama pull nomic-embed-text  # Pull required embedding model
```

### Clone the Repository
```
git clone https://github.com/Aayush212/Retrieval-Augmented-Generation-RAG-with-Streamlit-LangChain-FAISS-Ollama.git
cd Retrieval-Augmented-Generation-RAG-with-Streamlit-LangChain-FAISS-Ollama
```
### Set up Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Set up Environment Variables
Create a ```.env``` file in the project root and add your Groq API Key:
```
GROQ_API_KEY=your_groq_api_key
```
## Running the Application
```
streamlit run app.py
```
Once running, open the local URL displayed in the terminal (e.g., ```http://localhost:8501```).

## Usage

1. Upload research papers in PDF format to the ```research_papers```/ directory.

2. Click the "Document Embedding" button to process documents and create vector embeddings.

3. Enter a query in the text box and get AI-powered answers based on the research papers.

4. View retrieved document snippets under the "Document Similarity Search" section.

## File Structure
```
rag-app/
│── research_papers/       # Directory for storing research PDFs
│── app.py                 # Main Streamlit app
│── requirements.txt        # Python dependencies
│── .env                    # API keys (not to be committed)
│── README.md               # Documentation
```

## Troubleshooting

If you see ```ValueError: model "llama2" not found```, ensure you’ve pulled the correct Ollama embedding model:
```
ollama pull nomic-embed-text
```
If embeddings fail, check that your ```research_papers```/ folder contains valid PDFs.

## Future Enhancements

1. Support for multiple embedding models

2. UI improvements with advanced filters

3. Integration with a cloud-based vector database (e.g., Pinecone)

## License

This project is open-source under the MIT License. Feel free to use and improve it!

