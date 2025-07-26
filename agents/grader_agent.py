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
def grader_agent(state):
    docs = state.get("docs", [])
    if not docs:
        return {"docs": []}
    
    filtered = []
    for doc in docs:
        prompt = f"Is this document relevant to the question: '{state['question']}'?\n\n{doc.page_content}"
        response = llm.predict(prompt)
        if "yes" in response.lower():
            filtered.append(doc)
    return {"docs": filtered}