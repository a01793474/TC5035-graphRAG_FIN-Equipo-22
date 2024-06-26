import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from neo4j.exceptions import ClientError
from time import sleep

from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

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

# Query to fetch all company names from the graph
company_query = """
MATCH (c:Company)
RETURN c.name AS company_name
"""

# Function to fetch company names from the graph
def fetch_company_names(graph):
    try:
        result = graph.query(company_query)
        if result and len(result) > 0:
            company_names = [record["company_name"] for record in result]
            return company_names
        else:
            return []
    except Exception as e:
        print(f"Error fetching company names: {e}")
        return []

# Fetch company names from the graph database
COMPANY_NAMES = fetch_company_names(graph)
print(f"Company Names: {COMPANY_NAMES}")

# Function to extract company names from user input
def extract_company_names(user_input):
    mentioned_companies = [name for name in COMPANY_NAMES if name.lower() in user_input.lower()]
    print(f"Mentioned Companies: {mentioned_companies}")
    return mentioned_companies

# Function to retrieve folder metadata based on parent ID
@st.cache_data
def retrieve_folder_metadata(parent_id):
    folder_query = """
    MATCH (parent:Parent {id: $parent_id})-[:GRAND_FATHER]->(folder:Folder)
    RETURN coalesce(folder.name, 'Unknown Folder') AS folder_name
    """
    try:
        result = graph.query(folder_query, params={"parent_id": parent_id})
        if result and len(result) > 0:
            folder_names = [record["folder_name"] for record in result]
            return folder_names
        else:
            return ["Unknown Folder"]
    except Exception as e:
        print(f"Error retrieving folder name for parent ID {parent_id}: {e}")
        return ["Unknown Folder"]

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

        # Extract company names from user input
        mentioned_companies = extract_company_names(user_input)
        if not mentioned_companies:
            return [{"result": "No mentioned companies in the query."}]

        company_results = {}
        source_info = []

        # Process each company separately
        for company in mentioned_companies:
            company_query = f"""
            MATCH (folder:Folder)<-[:GRAND_FATHER]-(parent:Parent)-[:HAS_CHILD]->(node)
            WHERE folder.name STARTS WITH '{company}'
            RETURN parent.text AS text, score, parent.id AS parent_id_string, {{parent_id_string: parent.id}} AS metadata LIMIT 1
            """
            print(f"Running query for company: {company}")
            print(company_query)

            parent_vectorstore = Neo4jVector.from_existing_index(
                embeddings,
                index_name="parent_document",
                retrieval_query=company_query,
            )
            vector_parent = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=parent_vectorstore.as_retriever()
            )

            # Create a separate query for each company
            company_input = f"{user_input} for {company}"
            result = vector_parent.invoke(company_input) if vector_parent else [{"result": "Parent retriever is not initialized"}]

            if isinstance(result, str):
                result = [{"result": result, "metadata": {}}]
            elif isinstance(result, dict):
                result = [result]
            elif not isinstance(result, list):
                result = [{"result": str(result), "metadata": {}}]

            # Process results for the current company
            for res in result:
                company_results[company] = res.get("result", "")
                metadata = res.get("metadata", {})
                parent_id = metadata.get("parent_id_string", "Unknown")
                folder_names = retrieve_folder_metadata(parent_id)
                source_info.extend([f"{folder_name} Section_ID: {parent_id}" for folder_name in folder_names])

        # Prepare the context for the final answer
        context = "\n".join([f"{company}: {result}" for company, result in company_results.items()])
        sources = ", ".join(source_info)
        
        # Debugging: Print the context being fed into the final answer generation
        print("Context being fed to final answer generation:")
        print(context)

        # Define the prompt template
#         prompt_template = ChatPromptTemplate(
#             template="""
#             You are an assistant that helps consolidate information from multiple companies. Based on the following context, provide a logical and concise answer to the user's query.

#             Context:
#             {context}

#             User's Query:
#             {user_input}

#             Provide a consolidated answer:
#             """,
#             input_variables=["context", "user_input"]
#         )

#         # Create the LLM chain
#         final_answer_chain = LLMChain(
#             llm=llm,
#             prompt_template=prompt_template
#         )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an assistant that helps consolidate information from multiple companies. Based on the context you are receiving, provide a logical "
                        "and concise answer to the user's query. You will dismiss the company on which the text states there's no information availble "
                        "but must work you answer out with the company or companies on which you have received specific facts or information"
                        "If specific information is missing, highlight what is missing and how it affects the comparison.\n"
                        "Context:"
                        "{context}"



                    ),
                ),
                (
                    "human",
                    ("Provide a consolidated answer on the following user's query per the Context provided: {user_input}"),
                ),
            ]
        )

        final_answer_chain =  prompt_template | llm        
        
        # Generate the final answer
        final_answer_response = final_answer_chain.invoke({"context": context, "user_input": user_input})
        
        final_answer_content = final_answer_response.content if hasattr(final_answer_response, 'content') else final_answer_response

        answer_with_sources = f"{final_answer_content} (Source: 10K_Report {sources})"
        
         # Debugging: Print the final answer content
        print("Final answer content:")
        print(final_answer_content)

        return [{"result": answer_with_sources}]
    except Exception as e:
        print(f"Error during query_graph: {e}")
        return [{"result": f"Error processing the request: {str(e)}"}]
    
# Define the submit function
def submit():
    user_input = st.session_state.input
    st.session_state.user_msgs.append(user_input)
    st.session_state.start_time = datetime.now()  # Record the start time

    try:
        # Get the answer using the modified query_graph function
        result = query_graph(user_input)
        print(f"Debug result: {result}")

        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("result", "")
            st.session_state.system_msgs.append(answer)
            st.session_state.last_result = answer
        else:
            st.session_state.system_msgs.append("No result found")
            st.session_state.last_result = "No result found"
    except Exception as e:
        result = {"result": f"Error processing the request: {str(e)}"}
        st.session_state.system_msgs.append(result["result"])
        st.session_state.last_result = result["result"]

    st.session_state.input = ""
    st.session_state.question_asked = True

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