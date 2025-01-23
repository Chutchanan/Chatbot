import streamlit as st
import openai
import chromadb
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from openai import OpenAI

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

# Initialize the Persistent ChromaDB client (without path argument)
persist_directory = "D:/Project/Chatbot/chromadb_data2"
persistentClient = chromadb.PersistentClient(path=persist_directory)

# Retrieve the collection
collection_name = "document_embeddings"
collection = persistentClient.get_collection(collection_name)

# Initialize memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Generate embeddings using OpenAI API
def generate_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Or the appropriate embedding model
        input=text
    )
    return response.data[0].embedding

# Function to query ChromaDB for relevant documents
def query_chromadb(query, n_results=5):
    """Query ChromaDB collection for the most relevant results based on the query."""
    # Generate embedding for the query
    query_embedding = generate_embeddings(query)
    
    # Query the collection for most relevant results
    results = collection.query(
        query_embeddings=[query_embedding],  # Query with embedding
        n_results=n_results  # Limit to the top 'n' results
    )
    
    return results

# Function to interact with GPT-4 and generate a response
def generate_response(user_query, n_results=3):
    # Step 1: Query ChromaDB for relevant documents
    results = query_chromadb(user_query)

    # Step 2: Format the documents and metadata for GPT-4
    context = results['documents']  # Use only the document text as context

    # Add user query and retrieved context to memory
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(f"Retrieved Context: {context}")  
    
    # Print the context and the query before sending it to OpenAI
    print("Sending to OpenAI:")
    print("Context:\n", context)

    # Step 3: Prepare chat history for GPT-4
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Step 4: Generate the GPT-4 response using the context and chat history
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to help user answer questions about iCONEXT Company."},
            {"role": "user", "content": f"{user_query}\n\nContext:\n{context}\n\nHelp answer the question by using the context and your knowledge."},
            {"role": "assistant", "content": f"Conversation history:\n{chat_history}"}
        ]
    )
    
    # Step 5: Add GPT-4 response to memory
    ai_response = response.choices[0].message.content
    memory.chat_memory.add_ai_message(ai_response)
    return ai_response

# Streamlit app
# Initialize session state for chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store the conversation history

st.title("iCONEXT Company Assistant")
st.write("Ask questions about iCONEXT, and get instant answers powered by AI.")

# Input from user
user_query = st.text_input("Enter your query:")

# Handle query submission
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Fetching relevant documents and generating response..."):
            # Call your generate_response function
            response = generate_response(user_query)

            # Add the user query and response to chat history
            st.session_state.chat_history.append({"user": user_query, "bot": response})

        st.success("Here's the response:")
        st.write(response)
    else:
        st.warning("Please enter a query to proceed.")

# Display chat history
st.write("### Chat History")
if st.session_state.chat_history:
    for idx, chat in enumerate(st.session_state.chat_history):
        st.write(f"**You**: {chat['user']}")
        st.write(f"**Assistant**: {chat['bot']}")
else:
    st.write("No chat history yet.")
