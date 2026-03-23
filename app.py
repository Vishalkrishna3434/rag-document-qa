import streamlit as st  # builds the web UI (buttons, file uploader, text input)
from PyPDF2 import PdfReader  # reads and extracts text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits text into smaller chunks
import os  # used to access environment variables like API key

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # converts text chunks into vectors (numbers)
import google.generativeai as genai  # used to configure/authenticate with Gemini API
from langchain.vectorstores import FAISS  # stores vectors and searches them by meaning
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chatbot that generates the final answer
from langchain.chains.question_answering import load_qa_chain  # pipeline that combines chunks + question → answer
from langchain.prompts import PromptTemplate  # lets you customize instructions given to Gemini
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
    chunks=text_splitter.split_text(chunks)
    return chunks
  
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("false_index")

def get_conversational_chain():
   promp_template="""
   Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
   the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
   Context:\n {context}?\n
   Question:\n {question}\n
   
   Answer:
   """
   model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
   
   prompt=PromptTemplate(template=promp_template,input_variables=["context","question"])
   chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
   return chain
 
def user_input(user_question):
   embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
   

