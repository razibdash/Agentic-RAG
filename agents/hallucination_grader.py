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

def hallucination_grader(state):
    hallucination_score_prompt = f"""
    Evaluate if this content below includes hallucination for the question:\n
    Question: {state['question']}\n
    Context: {state.get('context', '')}\n
    Answer: {state.get('answer', '')}\n
    Respond only 'YES' or 'NO'.
    """
    result = llm.predict(hallucination_score_prompt)
    return {"hallucinated": result.strip().upper() == "YES"}