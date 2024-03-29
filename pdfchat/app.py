import os
import time
import tempfile
import pdfplumber
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain import hub
from streamlit_chat import message as chat_message
from langchain_community.document_loaders import YoutubeLoader
import re

# Function to record time
time_start = 0
def record_timing():
    global time_start
    if time_start != 0:
        duration = time.time() - time_start
        print(f"Time taken for query-response pair: {duration:.2f} seconds")
    time_start = time.time()

def is_youtube_link(url):
  youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([\w-]+)"
  return re.match(youtube_pattern, url) is not None

# List of available LLM models
llm_models = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Gemma-7B (Gated Access)": "google/gemma-7b",
    "Llama-2-7B (Gated Access)": "meta-llama/Llama-2",
    "Gemma-7B-it (Gated Access)":"google/gemma-7b-it",
    "Zephyr":"HuggingFaceH4/zephyr-7b-beta"
}

# Frontend code
st.title("Chat with PDF or URL")
st.write("**Upload your Hugging Face API token and either upload a PDF file or enter a URL below**", 
         unsafe_allow_html=True, 
         format="markdown", 
         style={'font-size': '20px'})

with st.sidebar:
    
    # Add a div to contain the input field and label
    st.markdown("<div id='hf_token_input'></div>", unsafe_allow_html=True)

    # Add CSS to ensure the input field stays in the default place
    st.markdown("""
        <style>
            #hf_token_input {
                top: 5rem;
                position: fixed; 
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Add the input field for the Hugging Face API token
    HF_token = st.text_input("Enter your Hugging Face API token", type="password")
    
    # Display error message if the token is not entered
    if not HF_token:
        st.error("Please enter your Hugging Face API token.")
    
    # Changing from here 
    if HF_token:
        
        st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Choose Model</h2>",
                    unsafe_allow_html=True)
        
        # Add a dropdown menu to select LLM model
        selected_model = st.selectbox("Select LLM Model", list(llm_models.keys()))
        
        # Display warning for models with gated access
        if selected_model == "Llama-2-7B (Gated Access)" or selected_model == "Gemma-7B (Gated Access)" or selected_model == "Gemma-7B-it (Gated Access)" :
            st.warning("Access to this model requires authorization from Hugging Face.")
        
        # Allow user to choose between uploading PDF or entering URL
        file_or_url = st.radio("Choose Input Type", ("PDF File", "Website", "Youtube Link"))

        if file_or_url == "PDF File":
            uploaded_file = st.file_uploader('Upload your .pdf file', type="pdf")
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    content = PDFPlumberLoader(tmp_file.name).load()
        else:  # Input type is URL
            url = st.text_input("Enter the URL")
            if is_youtube_link(url):
                loader = YoutubeLoader.from_youtube_url(
                    url, add_video_info=True
                )
                content = loader.load()
            else:
                content = WebBaseLoader(url).load()

        st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                    unsafe_allow_html=True)
        max_length = st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
        temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)
        if st.button("Apply Settings"):
            pass  # You can add your logic here if needed

    if 'content' in locals():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunking = text_splitter.split_documents(content)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(chunking, embeddings)
        prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

        # Get the selected LLM model ID
        selected_model_id = llm_models[selected_model]

        def model(user_query, max_length, temp):
            record_timing()  # Record time before generating response
            llm = HuggingFaceHub(
                repo_id=selected_model_id,
                huggingfacehub_api_token=HF_token,
                model_kwargs={"max_length": max_length, "temperature": temp}
            )
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
            qa = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
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

        # Reset chat when the model selection changes
        if "selected_model" in st.session_state:
            if st.session_state.selected_model != selected_model:
                st.session_state.messages = []
        st.session_state.selected_model = selected_model

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
            chat_message(message["content"], is_user=message["role"] == "user", key=message["content"])  # Assign unique key

        if user_prompt := st.text_input("Your message here", on_change=submit, key="text_input"):  # Assign unique key

            st.session_state.messages.append(
                {"role": "user", "content": user_prompt}
            )
            chat_message(user_prompt, is_user=True, key=user_prompt)  # Assign unique key
            response = model(user_prompt, max_length, temp)
            record_timing()  # Record time after generating response

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            chat_message(response, key=response)  # Assign unique key
