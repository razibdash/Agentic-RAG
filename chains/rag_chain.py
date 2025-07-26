from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGroq
from vectorstore.store import get_vectorstore
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,
    api_key=GROQ_API_KEY,
)
def get_rag_chain():
    retriever = get_vectorstore()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
