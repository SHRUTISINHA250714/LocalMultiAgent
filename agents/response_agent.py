import ollama

MODEL_NAME = "llama3.2:1b"

def response_agent(reasoning, query):
    """
    Generates a medium-length final answer based on reasoning.
    """
    truncated_reasoning = reasoning[:1200]

    messages = [
        {
            "role": "system",
            "content": "You are an expert assistant that gives clear and concise answers."
        },
        {
            "role": "user",
            "content": (
                f"Based on the following reasoning:\n{truncated_reasoning}\n\n"
                f"Question: {query}\n\n"
                "Instructions:\n"
                "- Provide a medium-length, coherent answer.\n"
                "- Include key points from the reasoning.\n"
                "- Avoid unnecessary repetition or overly long elaboration.\n"
                "- Expand only lightly with examples, types, or characteristics if needed."
            )
        }
    ]

    response = ollama.chat(model=MODEL_NAME, messages=messages)
    return response["message"].content
