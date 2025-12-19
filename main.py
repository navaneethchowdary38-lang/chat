import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# --------------------------------------------------
# PDF FUNCTIONS
# --------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# --------------------------------------------------
# QA CHAIN (FIXED)
# --------------------------------------------------
def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain


def user_input(user_question):
    vectorstore = load_faiss_index()
    qa_chain = get_qa_chain(vectorstore)
    response = qa_chain.invoke({"query": user_question})
    st.write("### ✅ Answer")
    st.write(response["result"])

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📘",
    layout="wide"
)

with st.sidebar:
    st.title("📂 Upload PDF")
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Analyze"):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            st.success("Vector database created successfully!")

def main():
    st.title("📄 LLM PDF Analyzer")
    st.subheader("Chat with your PDFs using Gemini + LangChain")

    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
