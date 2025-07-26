from langgraph.graph import StateGraph, END
from agents import router_agent, retriever_agent, grader_agent, hallucination_grader, answer_grader
from chains.tavily_chain import run_web_search

graph = StateGraph()

graph.add_node("router", router_agent.router_agent)
graph.add_node("retriever", retriever_agent.retriever_agent)
graph.add_node("grader", grader_agent.grader_agent)
graph.add_node("hallucination", hallucination_grader.hallucination_grader)
graph.add_node("answer_grader", answer_grader.answer_grader)
graph.add_node("web_search", lambda state: {"answer": run_web_search(state["question"])})

graph.set_entry_point("router")

graph.add_conditional_edges("router", lambda x: x["route"], {
    "retriever": "retriever",
    "web_search": "web_search"
})

graph.add_edge("retriever", "grader")
graph.add_edge("grader", "hallucination")
graph.add_edge("hallucination", "answer_grader")
graph.add_edge("web_search", "answer_grader")
graph.add_edge("answer_grader", END)

compiled_graph = graph.compile()
