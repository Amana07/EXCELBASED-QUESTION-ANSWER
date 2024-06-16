import os
import streamlit as st
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import nltk
import PyPDF2
import re
from nltk.tokenize import word_tokenize
import requests
import pandas as pd

# Download NLTK data
nltk.download('punkt')

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

def process_excel_file(file_path):
    text = ""
    try:
        df = pd.read_excel(file_path)
        for column in df.columns:
            text += " ".join(df[column].astype(str).values)
    except Exception as e:
        print(f"Error processing Excel file: {e}")
    cleaned_text = clean_text(text)
    return cleaned_text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
    return " ".join(cleaned_tokens)

def limit_text(text, max_words):
    tokens = word_tokenize(text)
    num_words = len(tokens)

    if num_words <= max_words:
        return text

    limited_tokens = tokens[:max_words]
    limited_text = " ".join(limited_tokens)
    return limited_text

def process_files(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    excel_files = [f for f in os.listdir(folder_path) if f.endswith((".xlsx", ".xls"))]
    documents = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pdf_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(pdf_text)
        documents.append(cleaned_text)
        
    for excel_file in excel_files:
        excel_path = os.path.join(folder_path, excel_file)
        excel_text = process_excel_file(excel_path)
        documents.append(excel_text)

    return documents

# Specify the folder path where the PDF and Excel files are located
folder_path = "book"

# Process the folder and get the extracted and cleaned text from each file
extracted_texts = process_files(folder_path)

# Join the elements of the extracted_texts list with newline characters to create a single string
output_text = "\n".join(extracted_texts)

# Save the extracted, cleaned, and limited text into a text file
output_file = "documents.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(output_text)

# Load the documents
documents = TextLoader('documents.txt').load()

def split_docs(documents, chunk_size=400, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

document_chunk = split_docs(documents)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(document_chunk, embeddings)

def main():
    st.title("PDF and Excel Text Extraction and Question Answering")
    
    # Sidebar
    folder_path = st.sidebar.text_input("Enter Folder Path", "book")

    # File upload
    pdf_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    excel_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if pdf_file:
        # Save the uploaded file to the specified folder
        upload_path = os.path.join(folder_path, pdf_file.name)
        with open(upload_path, "wb") as f:
            f.write(pdf_file.read())
        st.success(f"Uploaded file '{pdf_file.name}' saved to '{upload_path}'.")

    if excel_file:
        # Save the uploaded file to the specified folder
        upload_path = os.path.join(folder_path, excel_file.name)
        with open(upload_path, "wb") as f:
            f.write(excel_file.read())
        st.success(f"Uploaded file '{excel_file.name}' saved to '{upload_path}'.")

    if st.sidebar.button("Process Files"):
        extracted_texts = process_files(folder_path)
        output_text = "\n".join(extracted_texts)
        output_file = "documents.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(output_text)
        st.success(f"Extracted text from files in the folder has been saved to '{output_file}'.")

    # Main content
    query = st.text_input("Enter your question:")
    change_language = st.checkbox("Change the language of the response?")

    if change_language:
        new_language = st.text_input("Enter the desired language:")

    if st.button("Get Answer"):
        embedding_vector = embeddings.embed_query(query)
        docs = db.similarity_search_by_vector(embedding_vector)

        if not docs:
            st.warning("No relevant documents found.")
        else:
            message = f"{query}\nanswer on the basis of the documents\n{docs}if not in docs then say 'not in docs'".strip()

            # Use Google Generative Language API to get the desired text
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": message}]}]}

            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()

            if 'candidates' in response_data and response_data['candidates']:
                desired_text = response_data['candidates'][0]['content']['parts'][0]['text']
                st.write("Response:", desired_text)

                if change_language:
                    message1 = f"{query}\nanswer on the basis of docs\n{docs} if present in docs translate into {new_language}".strip()
                    # Use Google Generative Language API to get the desired text
                    data = {"contents": [{"parts": [{"text": message1}]}]}

                    response = requests.post(url, headers=headers, json=data)
                    response_data = response.json()

                    if 'candidates' in response_data and response_data['candidates']:
                        desired_text = response_data['candidates'][0]['content']['parts'][0]['text']
                        st.write("Response:", desired_text)

if __name__ == "__main__":
    main()
