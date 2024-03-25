import os
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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
HF_token='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'

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

pdf_file_path = None  # Initialize pdf_file_path

# Function to extract data from PDF
def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None  # You can build more code here to return a dataframe

# Frontend code
st.title("Chat with PDF")
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ Chat with PDF </h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h2>",
                unsafe_allow_html=True)
    st.markdown("Users can now add their own choice of PDFs and chat with our application")
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None:
        pdf_file_path = uploaded_file.name  # Set the pdf_file_path when file is uploaded

    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                unsafe_allow_html=True)
    max_length = st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)


# Set up the model and related components using the selected PDF file
if pdf_file_path:
    loader = PDFPlumberLoader(pdf_file_path)
    pages = loader.load()
    docs = text_splitter.split_documents(pages)
    db = Chroma.from_documents(docs, HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_token,model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    def model(user_query, max_length, temp):
        repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"max_length": 1024, "temperature": 0.1})
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=db.as_retriever(k=2),
                                         return_source_documents=True,
                                         verbose=True,
                                         chain_type_kwargs={"prompt": prompt})
        response = qa(user_query)["result"]
        # Extracting only the answer part
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
        response = model(user_prompt, max_length, temp)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        chat_message(response)
else:
    st.write("Please upload a PDF file to start.")
