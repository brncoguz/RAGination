# RAGination

**RAGination** is a chatbot project using Retrieval-Augmented Generation (RAG) and the Mistral AI client. The chatbot retrieves relevant context from a document and answers user questions intelligently.

Features:
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with generative AI.
- **Context-Aware QA:** Retrieves relevant text chunks to answer questions accurately.
- **Tool Usage:** Dynamically leverages tools to enhance responses.

Installation:
- Clone the project:
  - `git clone https://github.com/yourusername/RAGination.git`
- Install the requirements:
  - `cd RAGination`
  - `pip install -r requirements.txt`
- Set your Anthropic API key:
  - `export MISTRAL_API_KEY=your_api_key_here`
 
Usage:
- Run the chatbot:
  - `python rag.py`
- And start talking to it about anything. If you ask a question about AI, it'll use the document provided.
- Type `quit` to end the conversation.

Example Interaction:
```
Simple Chatbot (type 'quit' to exit)
You: Hello
Assistant: Hello! How can I assist you today?
You: What are the ways AI can mitigate climate change in transportation?
=====Mistral wants to use the qa_with_context tool=====
Assistant: Based on the provided context, AI can mitigate climate change in transportation by optimizing routes, improving vehicle efficiency, and promoting the adoption of electric vehicles. Here's how:

1. **Route Optimization**: AI can analyze traffic patterns and weather conditions in real-time to suggest the most fuel-efficient routes, reducing greenhouse gas emissions.

2. **Vehicle Efficiency**: AI can monitor vehicle performance and provide insights to drivers or fleet managers on how to improve fuel efficiency, such as adjusting speed, reducing idling time, or maintaining optimal tire pressure.

3. **Promoting Electric Vehicles (EVs)**: AI can help in the development and deployment of EVs by:
   - Optimizing battery design and charging infrastructure.
   - Predicting EV demand and usage patterns to inform infrastructure planning.
   - Encouraging the adoption of EVs through targeted marketing strategies based on AI-driven data analysis.
You: thanks
Assistant: You're welcome! If you have any more questions or need further assistance, feel free to ask.
You: bye
Assistant: Goodbye! Have a great day.
You: quit
Goodbye!
```

License:
- This project is for personal learning purposes and not for production use.
