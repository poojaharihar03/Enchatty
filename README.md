

## Abstract:
Arent we tired of using multiple OpenAI keys and their never-ending token limits so we have come up with our chatapp  Enchantty Streamlit-based application designed to streamline information retrieval through conversational interactions and deployed on streamlit cloud.

## Introduction:
Enchantty Chatbot introduces fresh possibilities to harness Hugging Face's open source language models and LangChain tools.Users may now upload PDF documents or enter URLs to extract important information via our chatbot. The chatbot responds to user queries with the most concise answers possible based on the content presented, allowing for quick access to specific information within documents or web pages. This program attempts to increase user productivity by automating the information extraction process, hence saving time and effort.

Enchantty Chatbot is a step toward more efficient and user-friendly information retrieval systems, catering to a wide range of user needs across multiple domains.
Here's the catch, you have to create a token with Hugging Face.

## Run Locally

Clone the project

```bash
  git clone https://github.com/anish2105/Enchatty.git
```

Create virtual environment

```bash
  python -m venv venv
  venv\Scripts\activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

## Run on Docker

Build Docker Image

```bash
  docker build -t test .
```
Run Your Image

```bash
  docker run -p 8501:8501 test
```

**Note:** You need to have docker desktop installed in your PC to run the above commands. 

# Screenshots
![image](https://github.com/anish2105/Enchatty/assets/88924201/8c823a71-c6fc-416e-82cd-127144398911)



