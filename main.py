import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local("faiss_index")


def load_vector_store():
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
# QA CHAIN (LANGCHAIN 1.x)
# --------------------------------------------------
def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4
    )

    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.
Use the provided context to answer the question.
If the answer is not found in the context, say "I don't know".

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


def answer_question(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and analyze PDFs first.")
        return

    vectorstore = load_vector_store()
    qa_chain = get_qa_chain(vectorstore)
    response = qa_chain.invoke({"query": user_question})

    st.markdown("### ✅ Answer")
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
    st.title("📂 Upload PDFs")
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Analyze PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                create_vector_store(chunks)
                st.success("Vector database created successfully!")

def main():
    st.title("📄 LLM PDF Analyzer")
    st.subheader("Chat with your PDFs using Gemini")

    question = st.text_input("Ask a question from the uploaded PDFs")

    if question:
        answer_question(question)

if __name__ == "__main__":
    main()

