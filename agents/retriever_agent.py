from chains.rag_chain import get_rag_chain

rag = get_rag_chain()

def retriever_agent(state):
    question = state["question"]
    return {"answer": rag.run(question)}
