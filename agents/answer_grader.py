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

def answer_grader(state):
    prompt = f"""
    Is the following answer accurate and grounded in the context?\n
    Question: {state['question']}\n
    Context: {state.get('context', '')}\n
    Answer: {state['answer']}\n
    Respond only 'YES' or 'NO'.
    """
    res = llm.predict(prompt)
    return {"final_decision": res.strip().upper() == "YES"}
