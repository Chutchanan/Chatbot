import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import openai
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

# # Initialize the Persistent ChromaDB client (without path argument)
# persist_directory = "/mount/src/chromadb_data2"
# persistentClient = chromadb.PersistentClient(path=persist_directory)

# # Retrieve the collection
# collection_name = "document_embeddings"
# collection = persistentClient.get_collection(collection_name)

configuration = {
    "client": "PersistentClient",
    "path": "/mount/src/chromadb_data2"
}

collection_name = "document_embeddings"

persistentClient = st.connection("chromadb",
                     type=ChromaDBConnection,
                     **configuration)
collection = conn.get_collection_data(collection_name)

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
    
    # Step 3: Generate the GPT-4 response using the retrieved context
    response = openai.chat.completions.create(  
        model="gpt-4o-mini",  # Specify the GPT-4o-mini model
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant to help user answer the question about iCONEXT Company"},
            {"role": "user", "content": f"{user_query}\n\nContext:\n{context}\n help to answer the question by using the context and your knowledge"}
        ]
    )
    
    # Accessing the content from the response object correctly
    return response.choices[0].message.content

# Streamlit app
st.title("iCONEXT Company Assistant")
st.write("Ask questions about iCONEXT, and get instant answers powered by AI.")

# Input from user
user_query = st.text_input("Enter your query:")

# Handle query submission
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Fetching relevant documents and generating response..."):
            response = generate_response(user_query)
        st.success("Here's the response:")
        st.write(response)
    else:
        st.warning("Please enter a query to proceed.")