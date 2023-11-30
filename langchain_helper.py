from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import  streamlit as st

load_dotenv()
google_api_key = os.environ['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key = google_api_key,temperature = 0.2)



embeddings = HuggingFaceInstructEmbeddings(
    query_instruction = "represent query for retrieval: "
)

file_path = "faiss_index"

def create_vector_db(pdf):
    pdfreader = PdfReader(pdf)
    # st.write(pdfreader)
    text = ""
    for page in pdfreader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    # print(chunks)

    vectordb = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectordb.save_local(file_path)



def get_qa_chain():
    vectordb = FAISS.load_local(file_path, embeddings)
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key = "query"
    )
    return chain

if __name__ == "__main__":

    # create_vector_db()
    chain = get_qa_chain()
    ans = chain("who is Atharv")
    print(ans['result'])
    # print(llm("how are you?"))



