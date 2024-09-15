
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import sys


# setup env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Initialize Gemini
llm = Gemini(model="models/gemini-pro")
embed_model = GeminiEmbedding(model_name="models/embedding-001")


# Set the LLM in the global settings
Settings.llm = llm
Settings.embed_model = embed_model

# query engine instance
query_engine = None


# Load data and convert to vector database
def load_data_and_create_vector_base():
    global query_engine
    docs = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    query_engine = index.as_query_engine()


def test_query():
    while True:
        query = input("Question (to exit type - exit) : ")
        if (query == "exit"):
            sys.exit()
        else:
            response = query_engine.query(query)
            print("Answer: ", response)


if __name__ == "__main__":
    load_data_and_create_vector_base()
    test_query()
