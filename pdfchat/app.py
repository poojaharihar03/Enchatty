import os
import time
import tempfile
import pdfplumber
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PDFPlumberLoader
from langchain import hub
from streamlit_chat import message as chat_message

# Set up Hugging Face API token
HF_token = st.sidebar.text_input("Enter your Hugging Face API token", type="password")
if not HF_token:
    st.error("Please enter your Hugging Face API token.")

if "max_length" not in st.session_state:
    st.session_state.max_length = 128  # Default value
if "temp" not in st.session_state:
    st.session_state.temp = 0.1  # Default value

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
pdf_file_path = None  # Initialize pdf_file_path

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None 

# Function to record time
time_start = 0
def record_timing():
    global time_start
    if time_start != 0:
        duration = time.time() - time_start
        print(f"Time taken for query-response pair: {duration:.2f} seconds")
    time_start = time.time()

# Frontend code
st.title("Chat with PDF")
st.write("**Upload your Hugging Face API token and PDF file below**", 
         unsafe_allow_html=True, 
         format="markdown", 
         style={'font-size': '30px'})
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ Chat with PDF </h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h2>",
                unsafe_allow_html=True)
    st.markdown("Users can now add their own choice of PDFs and chat with our application")
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if uploaded_file is not None:
    # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_file_path = tmp_file.name 
            
    # if uploaded_file is not None:
    #     pdf_file_path = uploaded_file.name  

    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                unsafe_allow_html=True)
    max_length = st.slider("Token Max Length", min_value=128, max_value=1024, value=st.session_state.max_length, step=128)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=st.session_state.temp, step=0.1)
    if st.button("Apply Settings"):
        st.session_state.max_length = max_length
        st.session_state.temp = temp


if pdf_file_path:
    loader = PDFPlumberLoader(pdf_file_path)
    pages = loader.load()
    docs = text_splitter.split_documents(pages)
    db = Chroma.from_documents(docs, HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_token, model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    def model(user_query, max_length, temp):
        record_timing()  # Record time before generating response
        repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        llm = HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=HF_token,
            model_kwargs={"max_length": 1024, "temperature": 0.1}
        )
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=db.as_retriever(k=2),
                                         return_source_documents=True,
                                         verbose=True,
                                         chain_type_kwargs={"prompt": prompt})
        response = qa(user_query)["result"]
       
        answer_start = response.find("Answer:")
        if answer_start != -1:
            answer = response[answer_start + len("Answer:"):].strip()
            return answer
        else:
            return "Sorry, I couldn't find the answer."

    # CSS styling for the text input
    styl = f"""
    <style>
        .stTextInput {{
            position: fixed;
            bottom: 3rem;
        }}
    </style>
    """
    st.markdown(styl, unsafe_allow_html=True)

    if "widget" not in st.session_state:
        st.session_state.widget = ''

    def submit():
        record_timing()  # Record time before submitting message
        st.session_state.something = st.session_state.widget
        st.session_state.widget = ''

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    for message in st.session_state.messages:
        chat_message(message["content"], is_user=message["role"] == "user")

    if user_prompt := st.text_input("Your message here", on_change=submit):

        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        chat_message(user_prompt, is_user=True)
        response = model(user_prompt, st.session_state.max_length, st.session_state.temp)
        record_timing()  # Record time after generating response

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        chat_message(response)

