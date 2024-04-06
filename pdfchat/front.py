import streamlit as st
from llm_code import *

# Frontend code
st.set_page_config(page_title="🦙💬 Enchantty Chatbot")
st.title("Enchantty")
st.write("Upload your Hugging Face API token and chat with your PDF's and Websites", 
         unsafe_allow_html=True, 
         format="markdown", 
         style={'font-size': '20px'})

with st.sidebar:
    st.markdown("*Getting a Hugging Face token*")
    st.markdown("Steps:")
    st.markdown("1. Navigate to [Hugging Face](https://huggingface.co/settings/tokens)")
    st.markdown("2. Create a write token and copy it to your clipboard")
    st.markdown("3. Paste the token in the input field below")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("*Hugging Face API Token*")
    token_placeholder = st.empty()
    HF_token = token_placeholder.text_input("Enter your Hugging Face API token", type="password")

    # Check if HF_token is provided
    if HF_token:
        # Replace the token input field with the success message
        token_placeholder.empty()
        st.success('API key provided!', icon='✅')
    else:
        st.warning('Please enter your Hugging Face API token!', icon='⚠️')

    # Changing from here 
    if HF_token:
        
        st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Choose Model</h2>",
                    unsafe_allow_html=True)
        
        # Add a dropdown menu to select LLM model
        selected_model = st.selectbox("Select LLM Model", list(llm_models.keys()))
        
        # Display warning for models with gated access
        if selected_model == "Llama-2-7B (Gated Access)" or selected_model == "Gemma-7B (Gated Access)" or selected_model == "Gemma-7B-it (Gated Access)" :
            st.warning("Access to this model requires authorization from Hugging Face.")
        
        file_or_url_placeholder = st.empty()
        file_or_url = st.radio("Choose Input Type", ("PDF File", "Website"))

        if file_or_url == "PDF File":
            uploaded_file = st.file_uploader('Upload your .pdf file', type="pdf")
            if uploaded_file is not None:
                # Replace the PDF upload input field with the success message
                file_or_url_placeholder.empty()
                st.success('PDF file uploaded successfully!', icon='✅')
                # Save the uploaded file to a temporary location and process it
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    content = PDFPlumberLoader(tmp_file.name).load()
                    
        elif file_or_url == "Website":
            url_placeholder = st.empty()
            url = st.text_input("Enter the URL")
            if url.strip():
                # Replace the URL input field with the success message
                url_placeholder.empty()
                st.success('URL entered successfully!', icon='✅')
                # Process the URL
                content = WebBaseLoader(url).load()

        st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                    unsafe_allow_html=True)
        max_length = st.slider("Token Max Length", min_value=256, max_value=1024, value=256, step=128)
        temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)


if 'content' in locals():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunking = text_splitter.split_documents(content)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunking, embeddings)
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
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    def submit():
        record_timing()  # Record time before submitting message
        st.session_state.something = st.session_state.widget
        st.session_state.widget = ''

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

    if user_prompt := st.chat_input("enter your query"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = model(user_prompt, max_length, temp)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
