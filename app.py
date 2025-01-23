import streamlit as st
import openai
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

# Initialize the Persistent ChromaDB client (without path argument)
persist_directory = "D:/Project/Chatbot/chromadb_data"
persistentClient = chromadb.PersistentClient(path=persist_directory)

# Retrieve the collection
collection_name = "website_embeddings"
retrieved_collection = persistentClient.get_collection(name=collection_name)

# Function to query ChromaDB for relevant documents
def query_chromadb(query, collection):
    # Generate the embedding for the query using OpenAI API
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",  # Embedding model
        input=query
    ).data[0].embedding
    
    # Query the collection for the closest documents based on cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=3  # Number of results to retrieve
    )
    
    # Debug: Print the raw structure of the results to understand its structure
    print("Query Results Structure:", results)

    # Extract documents and metadatas from the list of results
    # Assuming results is a list of dictionaries where each dictionary contains the relevant info
    documents = []
    metadatas = []

    # Check if results is a list and iterate through it to collect documents and metadata
    if isinstance(results, list):
        for result in results:
            documents.append(result.get('document', 'No document found'))
            metadatas.append(result.get('metadata', {}))  # Default to empty dict if no metadata found
    
    return documents, metadatas

# Function to interact with GPT-4 and generate a response
def generate_response(user_query):
    # Step 1: Query ChromaDB for relevant documents
    documents, metadatas = query_chromadb(user_query, retrieved_collection)
    
    # Step 2: Format the documents and metadata for GPT-4
    context = "\n".join([f"Title: {meta['Title']}\nDescription: {meta['Description']}\nURL: {meta['URL/Link']}\n" 
                         for meta in metadatas])
    
    # Step 3: Generate the GPT-4 response using the retrieved context
    response = openai.chat.completions.create(  
        model="gpt-4o-mini",  # Specify the GPT-4o-mini model
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant to help user answer the question about iCONEXT Company"},
            {"role": "user", "content": f"{user_query}\n\nContext:\n{context}"}
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