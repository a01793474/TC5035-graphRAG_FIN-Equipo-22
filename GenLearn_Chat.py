import os
import streamlit as st
from timeit import default_timer as timer
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from neo4j.exceptions import ClientError
from time import sleep

# Load from environment
load_dotenv('.env', override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Embeddings & LLM models
embedding_dimension = 1536
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3", api_version="2024-02-01", dimensions=embedding_dimension)

llm = AzureChatOpenAI(azure_deployment='chat_gtp_35', api_version="2023-05-15", temperature=0)

# Get Neo4j credentials from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    sleep(5)
except ClientError as e:
    print(f"Failed to connect to Neo4j: {e}")
    graph = None

from langchain_community.vectorstores import Neo4jVector

# Typical RAG retriever
try:
    typical_rag = Neo4jVector.from_existing_index(
        embeddings, index_name="typical_rag"
    )
except Exception as e:
    print(f"Failed to initialize typical RAG retriever: {e}")
    typical_rag = None

# Parent retriever
parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, parent.id AS parent_id_string, {parent_id_string: parent.id} AS metadata LIMIT 1
"""

try:
    parent_vectorstore = Neo4jVector.from_existing_index(
        embeddings,
        index_name="parent_document",
        retrieval_query=parent_query,
    )
except Exception as e:
    print(f"Failed to initialize parent retriever: {e}")
    parent_vectorstore = None

# Hypothetic questions retriever
hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, parent.id AS parent_id_string, {parent_id_string: parent.id} AS metadata
"""

try:
    hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
        embeddings,
        index_name="hypothetical_questions",
        retrieval_query=hypothetic_question_query,
    )
except Exception as e:
    print(f"Failed to initialize hypothetic questions retriever: {e}")
    hypothetic_question_vectorstore = None

# Summary retriever
summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, parent.id AS parent_id_string, {parent_id_string: parent.id} AS metadata
"""

try:
    summary_vectorstore = Neo4jVector.from_existing_index(
        embeddings,
        index_name="summary",
        retrieval_query=summary_query,
    )
except Exception as e:
    print(f"Failed to initialize summary retriever: {e}")
    summary_vectorstore = None

from langchain.chains import RetrievalQA

try:
    vector_typrag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=typical_rag.as_retriever()
    )
except Exception as e:
    print(f"Failed to initialize vector_typrag: {e}")
    vector_typrag = None

try:
    vector_parent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=parent_vectorstore.as_retriever()
    )
except Exception as e:
    print(f"Failed to initialize vector_parent: {e}")
    vector_parent = None

try:
    vector_hypquestion = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hypothetic_question_vectorstore.as_retriever()
    )
except Exception as e:
    print(f"Failed to initialize vector_hypquestion: {e}")
    vector_hypquestion = None

try:
    vector_summary = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=summary_vectorstore.as_retriever()
    )
except Exception as e:
    print(f"Failed to initialize vector_summary: {e}")
    vector_summary = None

def retrieve_metadata(user_input, retriever):
    try:
        print("Retrieving metadata with input:", user_input)  # Debugging print statement

        result = None
        if retriever == "Typical RAG retriever" and typical_rag:
            result = typical_rag.similarity_search_with_score(user_input, k=1)
        elif retriever == "Parent retriever" and parent_vectorstore:
            result = parent_vectorstore.similarity_search_with_score(user_input, k=1)
        elif retriever == "Hypothetic questions retriever" and hypothetic_question_vectorstore:
            result = hypothetic_question_vectorstore.similarity_search_with_score(user_input, k=1)
        elif retriever == "Summary retriever" and summary_vectorstore:
            result = summary_vectorstore.similarity_search_with_score(user_input, k=1)
        else:
            print(f"{retriever} is not initialized")
            return None

        print("Raw retrieval result:", result)  # Debugging print statement

        if result and isinstance(result, list) and len(result) > 0:
            metadata = result[0][0].metadata
            print(f"Metadata retrieved: {metadata}")
            return metadata
        else:
            print("No metadata retrieved")
            return None
    except Exception as e:
        print(f"Error during metadata retrieval: {e}")
        return None

def query_graph(user_input, retriever):
    try:
        print("Invoking vector function with input:", user_input)  # Debugging print statement

        result = None
        if retriever == "Typical RAG retriever" and vector_typrag:
            result = vector_typrag.invoke(user_input)
        elif retriever == "Parent retriever" and vector_parent:
            result = vector_parent.invoke(user_input)
        elif retriever == "Hypothetic questions retriever" and vector_hypquestion:
            result = vector_hypquestion.invoke(user_input)
        elif retriever == "Summary retriever" and vector_summary:
            result = vector_summary.invoke(user_input)
        else:
            print(f"{retriever} is not initialized")
            return {"result": f"{retriever} is not initialized"}

        print("Result from vector function:", result)  # Debugging print statement
        return result
    except Exception as e:
        print(f"Error during query_graph: {e}")
        return {"result": f"Error processing the request: {str(e)}"}

st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []
if "input" not in st.session_state:
    st.session_state.input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "last_time_taken" not in st.session_state:
    st.session_state.last_time_taken = 0.0
if "question_asked" not in st.session_state:
    st.session_state.question_asked = False
if "retriever" not in st.session_state:
    st.session_state.retriever = "Typical RAG retriever"  # Default retriever

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("GraphRAG - GenLearn for 10K SEC Data")
with img_col:
    st.image("./GenLearn_logo.jpg", width=350)

# Adding the live link
st.markdown("[Source URL](https://github.com/a01793474/TC5035-graphRAG_FIN-Equipo-22/tree/main/sec_10K_data/aapl-20230930)")

# Add a dropdown list for retriever selection
st.session_state.retriever = st.selectbox(
    "Select Retriever",
    ["Typical RAG retriever", "Parent retriever", "Hypothetic questions retriever", "Summary retriever"],
    index=["Typical RAG retriever", "Parent retriever", "Hypothetic questions retriever", "Summary retriever"].index(st.session_state.retriever)
)

def submit():
    user_input = st.session_state.input
    st.session_state.user_msgs.append(user_input)
    start = timer()

    try:
        # First, retrieve metadata separately
        metadata = retrieve_metadata(user_input, st.session_state.retriever)
        source_id = metadata.get("parent_id_string", "Unknown") if metadata else "Unknown"

        # Then, get the answer using RetrievalQA
        result = query_graph(user_input, st.session_state.retriever)
        print(f"Debug result: {result}")  # Debugging print statement
        answer = result["result"]
        answer_with_source = f"{answer} (Source ID: {source_id})"
        st.session_state.system_msgs.append(answer_with_source)
        st.session_state.last_result = answer_with_source
    except Exception as e:
        result = {"result": f"Error processing the request: {str(e)}"}
        st.session_state.system_msgs.append(result["result"])
        st.session_state.last_result = result["result"]

    st.session_state.input = ""  # Clear the input box
    st.session_state.last_time_taken = timer() - start
    st.session_state.question_asked = True

st.text_input("Enter your question", key="input", on_change=submit)

def message(text, is_user=False):
    if is_user:
        st.markdown(f"> **User:** {text}", unsafe_allow_html=True)
    else:
        st.markdown(f"> **GenLearn:** {text}", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

# Display the chat history
with col1:
    if st.session_state["user_msgs"]:
        st.markdown("### Questions")
        for i in range(len(st.session_state["user_msgs"]) - 1, -1, -1):
            message(st.session_state["user_msgs"][i], is_user=True)

with col2:
    if st.session_state["system_msgs"]:
        st.markdown("### Answers")
        for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
            message(st.session_state["system_msgs"][i], is_user=False)

# Display the time taken if a question has been asked
if st.session_state.question_asked:
    st.write(f"Time taken: {st.session_state.last_time_taken:.2f}s")
