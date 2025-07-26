def router_agent(state):
    question = state["question"]
    if any(keyword in question.lower() for keyword in ["latest", "current", "today", "now"]):
        return {"route": "web_search"}
    return {"route": "retriever"}
