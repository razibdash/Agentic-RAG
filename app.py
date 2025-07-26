from graph import compiled_graph

if __name__ == "__main__":
    question = input("Ask your question: ")
    result = compiled_graph.invoke({"question": question})
    print("\n✅ Final Result:", result.get("answer"))
