import os
import time
import tempfile
import pdfplumber
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain import hub
import re

# Function to record time
time_start = 0
def record_timing():
    global time_start
    if time_start != 0:
        duration = time.time() - time_start
        print(f"Time taken for query-response pair: {duration:.2f} seconds")
    time_start = time.time()

# List of available LLM models
llm_models = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Gemma-7B (Gated Access)": "google/gemma-7b",
    "Gemma-7B-it (Gated Access)":"google/gemma-7b-it",
    "Zephyr":"HuggingFaceH4/zephyr-7b-beta",
    "Google-2b (Gated Access)":"google/gemma-2b-it"
}

