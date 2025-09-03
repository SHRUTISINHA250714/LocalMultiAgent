import ollama

MODEL_NAME = "llama3.2:1b"

def reasoning_agent(docs, query):
    context = ""
    for d in docs:
        if len(context.split()) > 500:  
           break
        context += d.page_content[:500] + "\n\n"


    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant that explains topics clearly and concisely."
        },
        {
            "role": "user",
            "content": (
                f"Use the following documents to answer the question:\n\n"
                f"{context}\n\n"
                f"Question: {query}\n\n"
                "Instructions:\n"
                "- Include relevant info from the documents.\n"
                "- Expand with characteristics, types, examples and functions.\n"
                "- Keep the answer medium-length and informative, avoid unnecessary repetition."
            )
        }
    ]

    response = ollama.chat(model=MODEL_NAME, messages=messages)
    return response["message"].content
