import faiss
import functools
import os
import json
import numpy as np

from bs4 import BeautifulSoup
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Initialize Mistral client
def initialize_client():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the MISTRAL_API_KEY environment variable.")
    return MistralClient(api_key=api_key)

# Embedding functions
def get_text_embedding(client, txt):
    embeddings_batch_response = client.embeddings(model="mistral-embed", input=txt)
    return embeddings_batch_response.data[0].embedding

# QA function with context retrieval
def qa_with_context(client, text, question, chunk_size=512):
    def split_into_chunks(text, chunk_size):
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_into_chunks(text, chunk_size)
    text_embeddings = np.array([get_text_embedding(client, chunk) for chunk in chunks])

    # Build FAISS index
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Query embeddings and retrieve relevant chunks
    question_embeddings = np.array([get_text_embedding(client, question)])
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Construct prompt with retrieved context
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunks}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    response = client.chat(model="mistral-small-latest", messages=[ChatMessage(role="user", content=prompt)])
    return response.choices[0].message.content

# Tools configuration
def configure_tools(client, text):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "qa_with_context",
                "description": "Answer AI related user question by retrieving relevant context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "User question"},
                    },
                    "required": ["question"],
                },
            },
        }
    ]
    names_to_functions = {
        "qa_with_context": functools.partial(qa_with_context, client=client, text=text)
    }
    return tools, names_to_functions

# Chatbot interaction loop
def chatbot_loop(client, tools, names_to_functions):
    print("Simple Chatbot (type 'quit' to exit)")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # Add user message to history
        messages.append(ChatMessage(role="user", content=user_input))

        try:
            # Get response from Mistral
            response = client.chat(
                model="mistral-large-latest",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            # Handle tool calls or direct responses
            if response.choices[0].message.tool_calls is not None:
                tool_function = response.choices[0].message.tool_calls[0].function
                print(f"===== Mistral wants to use the {tool_function.name} tool =====")
                args = json.loads(tool_function.arguments)
                asst_message = names_to_functions[tool_function.name](**args)
            else:
                asst_message = response.choices[0].message.content

            # Print assistant's response and add to history
            print(f"Assistant: {asst_message}")
            messages.append(ChatMessage(role="assistant", content=asst_message))
        except Exception as e:
            print(f"An error occurred: {e}")

# Main function
def main():
    # Initialize Mistral client
    client = initialize_client()

    # Load text data
    file_name = "AI_greenhouse_gas.txt"
    with open(file_name, "r") as file:
        text = file.read()

    # Configure tools
    tools, names_to_functions = configure_tools(client, text)

    # Start chatbot
    chatbot_loop(client, tools, names_to_functions)

if __name__ == "__main__":
    main()
