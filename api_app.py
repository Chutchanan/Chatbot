from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')

# Ensure the API key is set correctly
if not api_key:
    raise Exception("API_KEY environment variable not set!")

# client = OpenAI(api_key=api_key)
openai.api_key = api_key

# Initialize the Persistent ChromaDB client
persist_directory = os.path.join(os.getcwd(), "chromadb_data2")
persistentClient = chromadb.PersistentClient(path=persist_directory)

# Retrieve the collection
collection_name = "document_embeddings"
try:
    collection = persistentClient.get_collection(collection_name)
except chromadb.errors.InvalidCollectionException:
    raise HTTPException(status_code=500, detail=f"Collection {collection_name} does not exist.")

# Initialize FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the iCONEXT API!"}

# Request model
class QueryRequest(BaseModel):
    query: str

# Generate embeddings using OpenAI API
def generate_embeddings(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",  # Or the appropriate embedding model
        input=text
    )
    return response.data[0].embedding

# Query ChromaDB for relevant documents
def query_chromadb(query, n_results=5):
    """Query ChromaDB collection for the most relevant results based on the query."""
    query_embedding = generate_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

# Generate a response using GPT-4
def generate_response(user_query, n_results=3):
    results = query_chromadb(user_query)
    context = results['documents']
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to help user answer the question about iCONEXT Company"},
            {"role": "user", "content": f"{user_query}\n\nContext:\n{context}\n help to answer the question by using the context and your knowledge"}
        ]
    )
    return response.choices[0].message.content

# Define the API endpoint
@app.post("/query")
async def handle_query(request: QueryRequest):
    user_query = request.query
    try:
        response = generate_response(user_query)
        return {"query": user_query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fix for the /favicon.ico request
@app.get("/favicon.ico")
async def favicon():
    return ""  # Empty response to prevent the error

# If running locally, use uvicorn to serve the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000)
