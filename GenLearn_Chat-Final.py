import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from neo4j.exceptions import ClientError
from time import sleep

# Configure the Streamlit page layout
st.set_page_config(layout="wide")

# Load environment variables from .env file
@st.cache_resource
def load_environment_variables():
    load_dotenv('.env', override=True)
    return {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    }

env_vars = load_environment_variables()

# Azure OpenAI credentials from environment variables
AZURE_OPENAI_ENDPOINT = env_vars["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = env_vars["AZURE_OPENAI_API_KEY"]

# Embeddings configuration
embedding_dimension = 1536
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3", api_version="2024-02-01", dimensions=embedding_dimension)

# Language model configuration
llm = AzureChatOpenAI(azure_deployment='Chat_gpt_4', api_version="2023-05-15", temperature=0)

# Neo4j database credentials from environment variables
NEO4J_URI = env_vars["NEO4J_URI"]
NEO4J_USERNAME = env_vars["NEO4J_USERNAME"]
NEO4J_PASSWORD = env_vars["NEO4J_PASSWORD"]

# Connect to Neo4j database
@st.cache_resource
def connect_to_neo4j(uri, username, password):
    try:
        graph = Neo4jGraph(url=uri, username=username, password=password)
        sleep(5)  # Give some time for the connection to stabilize
        return graph
    except ClientError as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

graph = connect_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

from langchain_community.vectorstores import Neo4jVector

# Parent retriever query 
parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
RETURN parent.text AS text, score, parent.id AS parent_id_string, {parent_id_string: parent.id} AS metadata LIMIT 1
"""

# Initialize Parent retriever
@st.cache_resource
def initialize_parent_retriever(_embeddings, parent_query):
    try:
        parent_vectorstore = Neo4jVector.from_existing_index(
            _embeddings,
            index_name="parent_document",
            retrieval_query=parent_query,
        )
        return parent_vectorstore
    except Exception as e:
        print(f"Failed to initialize parent retriever: {e}")
        return None

parent_vectorstore = initialize_parent_retriever(embeddings, parent_query)

from langchain.chains import RetrievalQA

# Initialize RetrievalQA for Parent retriever
@st.cache_resource
def initialize_retrieval_qa(_llm, _parent_vectorstore):
    try:
        vector_parent = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=_parent_vectorstore.as_retriever()
        )
        return vector_parent
    except Exception as e:
        print(f"Failed to initialize vector_parent: {e}")
        return None

vector_parent = initialize_retrieval_qa(llm, parent_vectorstore)

# Function to retrieve folder metadata based on parent ID
@st.cache_data
def retrieve_folder_metadata(parent_id):
    folder_query = """
    MATCH (parent:Parent {id: $parent_id})-[:GRAND_FATHER]->(folder:Folder)
    RETURN coalesce(folder.name, 'Unknown Folder') AS folder_name
    """
    try:
        # Run the query on the Neo4j graph
        result = graph.query(folder_query, params={"parent_id": parent_id})
        if result and len(result) > 0:
            folder_name = result[0]["folder_name"]
            return folder_name
        else:
            return "Unknown Folder"
    except Exception as e:
        print(f"Error retrieving folder name for parent ID {parent_id}: {e}")
        return "Unknown Folder"

# Function to retrieve parent metadata based on user input
def retrieve_parent_metadata(user_input):
    try:
        print("Retrieving parent metadata with input:", user_input)

        result = parent_vectorstore.similarity_search_with_score(user_input, k=10) if parent_vectorstore else None

        if result and isinstance(result, list) and len(result) > 0:
            metadata_list = [r[0].metadata for r in result if hasattr(r[0], 'metadata')]
            print(f"Parent metadata retrieved: {metadata_list}")
            return metadata_list
        else:
            print("No parent metadata retrieved")
            return None
    except Exception as e:
        print(f"Error during parent metadata retrieval: {e}")
        return None

# Function to query the graph based on user input
def query_graph(user_input):
    try:
        print("Invoking vector function with input:", user_input)

        result = vector_parent.invoke(user_input) if vector_parent else [{"result": "Parent retriever is not initialized"}]

        print("Result from vector function:", result)
        
        # Ensure result is a list of dictionaries
        if isinstance(result, str):
            result = [{"result": result, "metadata": {}}]
        elif isinstance(result, dict):
            result = [result]
        elif not isinstance(result, list):
            result = [{"result": str(result), "metadata": {}}]

        return result
    except Exception as e:
        print(f"Error during query_graph: {e}")
        return [{"result": f"Error processing the request: {str(e)}"}]

# Initialize session state variables
if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []
if "input" not in st.session_state:
    st.session_state.input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "question_asked" not in st.session_state:
    st.session_state.question_asked = False

# Create columns for the title and image
title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("GraphRAG - GenLearn for 10K SEC Data")
with img_col:
    st.image("./GenLearn_logo.jpg", width=350)

# Add a link to the source URL
st.markdown("[Source URL](https://github.com/a01793474/TC5035-graphRAG_FIN-Equipo-22/tree/main/sec_10K_data)")

# Define the submit function
def submit():
    user_input = st.session_state.input
    st.session_state.user_msgs.append(user_input)
    st.session_state.start_time = datetime.now()  # Record the start time

    try:
        # Retrieve parent metadata
        parent_metadata_list = retrieve_parent_metadata(user_input)
        source_info = []
        if parent_metadata_list:
            for metadata in parent_metadata_list:
                parent_id = metadata.get("parent_id_string", "Unknown")
                folder_name = retrieve_folder_metadata(parent_id)
                source_info.append(f"{folder_name} Section_ID: {parent_id}")
        else:
            source_info.append("Unknown Source")

        # Get the answer using RetrievalQA
        result = query_graph(user_input)
        print(f"Debug result: {result}")

        # Assume result is a single dictionary in a list
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("result", "")
            answer_with_sources = f"{answer} (Source: 10K_Report {', '.join(source_info)})"
            st.session_state.system_msgs.append(answer_with_sources)
            st.session_state.last_result = answer_with_sources
        else:
            st.session_state.system_msgs.append("No result found")
            st.session_state.last_result = "No result found"
    except Exception as e:
        result = {"result": f"Error processing the request: {str(e)}"}
        st.session_state.system_msgs.append(result["result"])
        st.session_state.last_result = result["result"]

    st.session_state.input = ""
    st.session_state.question_asked = True

# Input box for user questions
st.text_input("Enter your question", key="input", on_change=submit)

# Function to display messages in the chat
def message(text, is_user=False, time_taken=None):
    if is_user:
        st.markdown(f"> **User:** {text}", unsafe_allow_html=True)
    else:
        st.markdown(f"> **GenLearn:** {text}", unsafe_allow_html=True)
        if time_taken:
            st.write(f"Time taken: {time_taken.total_seconds():.2f}s")

# Create a single column for displaying the chat history
with st.container():
    if st.session_state["user_msgs"] and st.session_state["system_msgs"]:
        st.markdown("### Chat History")
        for i in range(len(st.session_state["user_msgs"]) - 1, -1, -1):
            message(st.session_state["user_msgs"][i], is_user=True)
            if i < len(st.session_state["system_msgs"]):
                if i == 0 and st.session_state.start_time:
                    end_time = datetime.now()  # Record the end time
                    time_taken = end_time - st.session_state.start_time
                    message(st.session_state["system_msgs"][i], is_user=False, time_taken=time_taken)
                    st.session_state.start_time = None  # Reset the start time
                else:
                    message(st.session_state["system_msgs"][i], is_user=False)
