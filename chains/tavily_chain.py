from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,
    api_key=GROQ_API_KEY,
)

search = TavilySearchResults(k=2)
tools = [Tool(name="Tavily", func=search.run, description="Search the web.")]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
def run_web_search(query):
    return agent.run(query)


result=run_web_search("What is the capital of France?")
print(result)