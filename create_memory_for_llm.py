# install the following before execution
#  pip install python-dotenv   
#  pip install langchain langchain_community langchain_huggingface faiss-cpu

# code to run python create_memory_for_llm.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
print("done")


DATA_PATH="data/"
def load_pdf_files(data):#loading data
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
# print("length of pdf:",len(documents))


# create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
# print("len of text chunks:",len(text_chunks))

# create vector embeddings
def get_embedding_model():# allow semantic search
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    return embedding_model 

embedding_model=get_embedding_model()

# store vector embeddings 
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)