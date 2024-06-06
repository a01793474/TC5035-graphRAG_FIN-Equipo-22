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
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
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
RETURN parent.text AS text, score, {} AS metadata
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
RETURN parent.text AS text, score, {} AS metadata
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

def query_graph(user_input):
    try:
        print("Invoking vector function with input:", user_input)  # Debugging print statement

        # Assuming vector_parent.invoke is a synchronous call and returns a result
        if vector_parent:
            result = vector_parent.invoke(user_input)
            print("Result from vector function:", result)  # Debugging print statement
            return result
        else:
            print("vector_parent is not initialized")
            return {"result": "vector_parent is not initialized"}
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

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("GraphRAG - GenLearn for 10K SEC Data")
with img_col:
    st.image("./GenLearn_logo.jpg", width=210)

def submit():
    user_input = st.session_state.input
    st.session_state.user_msgs.append(user_input)
    start = timer()

    try:
        result = query_graph(user_input)
        answer = result["result"]
        st.session_state.system_msgs.append(answer)
        st.session_state.last_result = answer
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
        st.markdown("### User Questions")
        for i in range(len(st.session_state["user_msgs"]) - 1, -1, -1):
            message(st.session_state["user_msgs"][i], is_user=True)

with col2:
    if st.session_state["system_msgs"]:
        st.markdown("### GenLearn Answers")
        for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
            message(st.session_state["system_msgs"][i], is_user=False)

# Display the time taken if a question has been asked
if st.session_state.question_asked:
    st.write(f"Time taken: {st.session_state.last_time_taken:.2f}s")
