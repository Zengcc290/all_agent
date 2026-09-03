from agents.llm import LLM

if __name__ == "__main__":
    llm = LLM()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{llm.get_query()}"},
    ]
    response = llm.think(messages)
    print("\n-----思考完成-----")
