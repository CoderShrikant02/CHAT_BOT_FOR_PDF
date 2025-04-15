#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Chatbot Backend Module - Updated with ChromaDB SQLite Fix

This module provides the necessary functions for implementing a 
Retrieval Augmented Generation (RAG) chatbot using Gemini API, 
LangChain framework, and ChromaDB as vector database.
"""

# Fix for ChromaDB sqlite3 compatibility issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Standard imports
import os
import pdfplumber
import google.generativeai as genai
from typing import Optional, List, Dict, Tuple, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import traceback


# ----------------------------------------
# Setup and Configuration
# ----------------------------------------

def setup_api_key(api_key: str) -> None:
    """
    Configure the Gemini API with the provided key.
    
    Args:
        api_key (str): Google API key for Gemini
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    print("API key configured successfully")


# ----------------------------------------
# Section 1: Uploading PDF
# ----------------------------------------

def upload_pdf(pdf_path: str) -> Optional[str]:
    """
    Function to handle PDF uploads.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: PDF file path if successful, None otherwise
    """
    try:
        if os.path.exists(pdf_path):
            print(f"PDF file found at: {pdf_path}")
            return pdf_path
        else:
            print(f"Error: File not found at {pdf_path}")
            return None
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None


# ----------------------------------------
# Section 2: Parsing the PDF
# ----------------------------------------

def parse_pdf(pdf_path: str) -> Optional[str]:
    """
    Function to extract text from PDF files.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text from the PDF, None if error
    """
    try:
        text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        
        print(f"PDF parsed successfully, extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        traceback.print_exc()
        return None


# ----------------------------------------
# Section 3: Creating Document Chunks
# ----------------------------------------

def create_document_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Function to split the document text into smaller chunks for processing.
    
    Args:
        text (str): The full text from the PDF
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks to maintain context
        
    Returns:
        List[str]: List of text chunks
    """
    try:
        if not text or len(text.strip()) == 0:
            print("Warning: Empty text provided to document chunker")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) > 0]
        
        print(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error creating document chunks: {e}")
        traceback.print_exc()
        return []


# ----------------------------------------
# Section 4: Embedding the Documents
# ----------------------------------------

def init_embedding_model(model_name: str = "models/text-embedding-004") -> Optional[GoogleGenerativeAIEmbeddings]:
    """
    Initialize the Gemini embeddings model.
    
    Args:
        model_name (str): Name of the embedding model to use
        
    Returns:
        Optional[GoogleGenerativeAIEmbeddings]: Embedding model for further use, None if error
    """
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name
        )
        print("Embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        traceback.print_exc()
        return None


def embed_documents(embedding_model: GoogleGenerativeAIEmbeddings, text_chunks: List[str]) -> bool:
    """
    Function to test the embedding model.
    
    Args:
        embedding_model: The embedding model to use
        text_chunks (List[str]): List of text chunks from the document
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if text_chunks and len(text_chunks) > 0:
            test_embedding = embedding_model.embed_query(text_chunks[0][:100])
            if test_embedding and len(test_embedding) > 0:
                print("Embedding test successful")
                return True
            else:
                print("Embedding test failed")
                return False
        else:
            print("No text chunks to embed")
            return False
    except Exception as e:
        print(f"Error testing embedding model: {e}")
        traceback.print_exc()
        return False


# ----------------------------------------
# Section 5: Storing in Vector Database (ChromaDB)
# ----------------------------------------

def store_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings, 
    text_chunks: List[str], 
    collection_name: str = "default_collection",
    persist_directory: str = "./chroma_db",
    metadatas: Optional[List[Dict[str, str]]] = None
) -> Optional[Chroma]:
    """
    Function to store document embeddings in ChromaDB with SQLite compatibility fixes.
    
    Args:
        embedding_model: The embedding model to use
        text_chunks (List[str]): List of text chunks to embed and store
        collection_name (str): Name of the collection in ChromaDB
        persist_directory (str): Directory to persist the database
        metadatas (Optional[List[Dict[str, str]]]): Metadata for each chunk
        
    Returns:
        Optional[Chroma]: Vector store for retrieval, None if error
    """
    try:
        print(f"Starting store_embeddings with {len(text_chunks)} chunks")
        
        # Validate inputs
        if not text_chunks or len(text_chunks) == 0:
            raise ValueError("No text chunks provided")
            
        # Handle metadata
        if metadatas is None:
            metadatas = [{"source": "document"} for _ in text_chunks]
        elif len(metadatas) != len(text_chunks):
            print("Warning: Mismatched metadata, using default")
            metadatas = [{"source": "document"} for _ in text_chunks]
        
        # Create persist directory if needed
        try:
            os.makedirs(persist_directory, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create directory {persist_directory}: {e}")
            persist_directory = None
        
        # Try with persistence first
        try:
            print("Attempting to create persistent Chroma collection...")
            vectorstore = Chroma.from_texts(
                texts=text_chunks,
                embedding=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
                metadatas=metadatas
            )
            if persist_directory:
                vectorstore.persist()
            return vectorstore
            
        except Exception as e:
            print(f"Persistent Chroma failed, trying in-memory: {e}")
            # Fallback to in-memory
            return Chroma.from_texts(
                texts=text_chunks,
                embedding=embedding_model,
                metadatas=metadatas
            )
            
    except Exception as e:
        print(f"Critical error in store_embeddings: {e}")
        traceback.print_exc()
        return None


# ----------------------------------------
# Section 6 & 7: Context Retrieval
# ----------------------------------------

def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n"):
    """
    Extract page_content from document chunks and join them with a splitter.
    
    Args:
        relevant_chunks (list): List of document chunks from retriever
        splitter (str): String to use as separator between chunk contents
        
    Returns:
        str: Combined context from all chunks
    """
    try:
        if not relevant_chunks or len(relevant_chunks) == 0:
            print("Warning: No relevant chunks found")
            return "No relevant information found in the documents."
    
        chunk_contents = []
        for i, chunk in enumerate(relevant_chunks):
            if hasattr(chunk, 'page_content'):
                chunk_text = f"[Chunk {i+1}]: {chunk.page_content}"
                chunk_contents.append(chunk_text)
        
        return splitter.join(chunk_contents)
    except Exception as e:
        print(f"Error combining chunks: {str(e)}")
        traceback.print_exc()
        return "Error retrieving context from documents."


# ----------------------------------------
# Section 8: Generating Responses with Gemini
# ----------------------------------------

def query_with_full_context(
    query: str, 
    vectorstore: Chroma, 
    model_name: str = "gemini-1.5-pro",
    k: int = 3,
    temperature: float = 0.3
) -> Tuple[str, str, List[Any]]:
    """
    Comprehensive query function with error handling and fallbacks.
    
    Args:
        query (str): User's question
        vectorstore: The ChromaDB vector store
        model_name (str): Name of the Gemini model to use
        k (int): Number of chunks to retrieve
        temperature (float): Temperature for response generation
        
    Returns:
        tuple: (response, context, chunks)
    """
    try:
        if not query or len(query.strip()) == 0:
            return "Please provide a valid question.", "", []
            
        print(f"Processing query: '{query}'")
        
        # 1. Retrieve relevant chunks
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            relevant_chunks = retriever.get_relevant_documents(query)
            print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        except Exception as retriever_error:
            print(f"Error retrieving chunks: {str(retriever_error)}")
            traceback.print_exc()
            return (f"Error retrieving information: {str(retriever_error)}", "", [])
        
        # 2. Get combined context
        context = get_context_from_chunks(relevant_chunks)
        
        # 3. Create enhanced prompt
        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Use ONLY the following context to answer the question. 
If the answer cannot be determined from the context, respond with "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        # 4. Initialize Gemini model
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=1024
            )
            
            # 5. Generate response
            response = llm.invoke(prompt)
            
            return response.content, context, relevant_chunks
        except Exception as model_error:
            print(f"Error with language model: {str(model_error)}")
            traceback.print_exc()
            
            # Try with a different model as fallback
            try:
                fallback_model = "gemini-1.0-pro"
                print(f"Trying fallback model: {fallback_model}")
                llm_fallback = ChatGoogleGenerativeAI(
                    model=fallback_model,
                    temperature=temperature,
                    max_output_tokens=1024
                )
                response = llm_fallback.invoke(prompt)
                return response.content, context, relevant_chunks
            except Exception as fallback_error:
                print(f"Fallback model also failed: {str(fallback_error)}")
                traceback.print_exc()
                return (f"Error generating response. Technical details: {str(model_error)}", context, relevant_chunks)
                
    except Exception as e:
        print(f"Error in query_with_full_context: {e}")
        traceback.print_exc()
        return f"Error generating response: {str(e)}", "", []
