import streamlit as st  # builds the web UI (buttons, file uploader, text input)
from PyPDF2 import PdfReader  # reads and extracts text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits text into smaller chunks
import os  # used to access environment variables like API key

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # converts text chunks into vectors (numbers)
import google.generativeai as genai  # used to configure/authenticate with Gemini API
from langchain_community.vectorstores import FAISS  # stores vectors and searches them by meaning
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chatbot that generates the final answer
from langchain_classic.chains.question_answering import load_qa_chain # pipeline that combines chunks + question → answer
from langchain_core.prompts import PromptTemplate  # lets you customize instructions given to Gemini
from dotenv import load_dotenv  # loads your .env file

load_dotenv()  # reads GOOGLE_API_KEY from .env file into the environment

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdfdocs):
  text=""
  for pdf in pdfdocs:
    pdf_reader=PdfReader(pdf)
    for page in pdf_reader.pages:
      text+=page.extract_text()
  return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks
  
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
   promp_template="""
   Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
   the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
   Context:\n {context}?\n
   Question:\n {question}\n
   
   Answer:
   """
   model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.3)
   
   prompt=PromptTemplate(template=promp_template,input_variables=["context","question"])
   chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
   return chain
 
def user_input(user_question):
   if not os.path.exists("faiss_index"):
    st.error("Please upload and process the PDF first.")
    return
   
   embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
   
   new_db= FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
   docs=new_db.similarity_search(user_question)
   
   chain=get_conversational_chain()
   
   response=chain(
     {"input_documents":docs,
      "question":user_question},
      return_only_outputs=True
   )
   
   print(response)
   st.write("Reply:",response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()