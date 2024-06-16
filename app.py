import os
import pandas as pd
import streamlit as st
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import nltk
import re
from nltk.tokenize import word_tokenize
import requests

# Download NLTK data
nltk.download('punkt')

def extract_text_from_excel(excel_file):
    text = ""
    xls = pd.ExcelFile(excel_file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        text += df.to_string(index=False, header=False)
    return text

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

def process_excel_folder(folder_path):
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") or f.endswith(".xls")]
    documents = []

    for excel_file in excel_files:
        excel_path = os.path.join(folder_path, excel_file)
        excel_text = extract_text_from_excel(excel_path)
        cleaned_text = clean_text(excel_text)
        documents.append(cleaned_text)

    return documents

# Specify the folder path where the Excel files are located
excel_folder_path = "book"

# Process the Excel folder and get the extracted and cleaned text from each Excel file
extracted_texts = process_excel_folder(excel_folder_path)

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
    st.title("Excel Text Extraction and Question Answering")
    
    # Sidebar
    excel_folder_path = st.sidebar.text_input("Enter Excel Folder Path", "book")

    # File upload
    excel_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if excel_file:
        # Save the uploaded file to the specified folder
        upload_path = os.path.join(excel_folder_path, excel_file.name)
        with open(upload_path, "wb") as f:
            f.write(excel_file.read())
        st.success(f"Uploaded file '{excel_file.name}' saved to '{upload_path}'.")

    if st.sidebar.button("Process Excels"):
        extracted_texts = process_excel_folder(excel_folder_path)
        output_text = "\n".join(extracted_texts)
        output_file = "documents.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(output_text)
        st.success(f"Extracted text from Excels in the folder has been saved to '{output_file}'.")

    # Main content
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        embedding_vector = embeddings.embed_query(query)
        docs = db.similarity_search_by_vector(embedding_vector)

        if not docs:
            st.warning("No relevant documents found.")
        else:
            message = f"{query}\n Please provide the correct answer based on the information found in \n{docs}if not in {docs}then say not in doc if question is what is products of apple then answer is i phone, ipad, mac ".strip()

            # Use Google Generative Language API to get the desired text
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyDUI__vq_DaIZRmJpebK2elYLbosaTXjUc"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": message}]}]}

            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()

            if 'candidates' in response_data and response_data['candidates']:
                desired_text = response_data['candidates'][0]['content']['parts'][0]['text']
                st.write("Response:", desired_text)

if __name__ == "__main__":
    main()

