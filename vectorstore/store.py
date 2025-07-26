from langchain.vectorstores import FAISS
from src.helper import download_hugging_face_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

embeddings=download_hugging_face_embeddings()

def get_vectorstore(path="../research/attenstion_is_all_you_need.pdf"):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    store = FAISS.from_documents(splits, embeddings)
    return store.as_retriever()
