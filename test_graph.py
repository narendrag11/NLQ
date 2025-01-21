import json
import logging
from typing import Annotated, Literal, TypedDict, List, Dict, Any, Optional, Tuple
from uuid import UUID
import getpass
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
import numpy as np
from sqlalchemy import create_engine, text,select
from sqlalchemy.orm import Session
from models import *
from dotenv import load_dotenv
import os



# Load environment variables from the .env file
load_dotenv()

# Check and get the API key from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"
# Initialize Database Engine
engine = create_engine(DATABASE_URL)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
        )
from typing import TypedDict, Dict, Any
class GraphConfig(TypedDict):
    engine: str
    limit: int

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_prompt: Optional[str]
    intent: Optional[Dict[str, Any]]
    embedding_vector: Optional[List[float]]
    domains_id: Optional[List[str]]
    datasource_ids: Optional[List[UUID]]
    selected_columns: Optional[List[Tuple[UUID, str, float]]]
    sql_query: Optional[str]
    sql_queries: Optional[List[str]]
    fallback: Optional[bool]
    status: Optional[str]
    config: GraphConfig
    #TODO 
    # add user_id, user_prompt, intent, workspace_id,domain_ids,status to db..
def intent_agent(state: State) -> Dict[str, List[AnyMessage]]:
    user_query = state.get("user_prompt", "")
    user_query="How much did I make last month for each store?"
    if not user_query:
        state["status"] = "No User Prompt"
        return {"messages": [AIMessage(content="No user prompt provided.")]}

    # Define the system instruction for the LLM
    system_instruction = (
        """You are an expert in Natural Language Understanding (NLU) and intent recognition. 
        Your task is to analyze user input and extract structured information."""
        + user_query
    )

    # JSON schema to structure the output
    json_schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"},
        },
        "required": ["intent"],
    }

    # Prepare LLM messages
    messages = [
        {"role": "system", "content": system_instruction},  # System message
        {"role": "user", "content": user_query},  # User query
    ]

    try:
        # Invoke the LLM
        llm_with_schema = llm.with_structured_output(json_schema)
        ai_msg = llm_with_schema.invoke(messages)

        # Extract structured intent
        data_to_embed = ai_msg[0]["args"]
        data_as_string = json.dumps(data_to_embed)

        # Generate embeddings
        intent_embedding = embeddings.embed_query(data_as_string)

        # Update the state
        state["intent"] = data_to_embed
        state["embedding_vector"] = intent_embedding

        # Create a message for extracted intent
        intent_message = AIMessage(content=f"Extracted intent: {data_to_embed}")
        return {"messages": [intent_message]}
    except Exception as e:
        logger.error(f"Intent Agent failed: {e}")
        state["status"] = "Intent Extraction Failed"
        error_message = AIMessage(content="Failed to extract intent. Please try again.")
        return {"messages": [error_message]}

def domain_agent(state: Dict[str, Any], config) -> Dict[str, List[AnyMessage]]:

    engine = config.get('configurable', {}).get('engine')
    limit = config.get('configurable', {}).get('limit', 5)
    query_intent_embedding = state.get("embedding_vector")
    try:
        with Session(engine) as session:
            stmt = (
                select(
                    DomainEmbedding,  
                    Domain.id.label("domain_id"),    
                    Domain.name.label("domain_name")  ,  
                    (1 - DomainEmbedding.embedding.cosine_distance(query_intent_embedding)).label("similarity_score")
                )
                .join(Domain, DomainEmbedding.domain_id == Domain.id)
                .order_by((1 - DomainEmbedding.embedding.cosine_distance(query_intent_embedding)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()
            
            # Process the results
            domains = []
            for domain_embedding, domain, similarity_score in results:
                domains.append({
                    "domain_id": str(domain.id),
                    "domain_name": domain.name,
                    "similarity_score": round(similarity_score, 4),
                })

            # Update state with retrieved domain IDs
            state["domain_ids"] = [domain["domain_id"] for domain in domains]

            # Create a success message
            domain_message = AIMessage(content=json.dumps({"domains": domains}, indent=2))
            print("domain_message ---",domain_message)
            return {"messages": [domain_message]}
    except Exception as e:
        error_message = AIMessage(content=f"Domain retrieval failed: {str(e)}")
        return {"messages": [error_message]}


def datasources_agent(state: Dict[str, Any],config) -> Dict[str, List[AIMessage]]:
    engine = config.get('configurable', {}).get('engine')
    limit = config.get('configurable', {}).get('limit', 5)
    domain_ids = state.get("domain_ids", [])
    query_embedding = state.get("embedding_vector", [])
    # Ensure required parameters are present
    if not domain_ids or not query_embedding or engine is None:
        error_message = AIMessage(content="Missing required parameters: domain_ids, query_embedding, or engine.")
        state["status"] = "Error"
        return {"messages": [error_message]}

    with Session(engine) as session:
        # Step 1: Retrieve Datasource IDs associated with the given Domain IDs
        datasource_ids_query = (
            select(DomainDatasource.datasource_id)
            .where(DomainDatasource.domain_id.in_(domain_ids))
        )
        datasource_ids = [row[0] for row in session.execute(datasource_ids_query).all()]

        if datasource_ids:
            # Step 2: Perform similarity search on TableEmbedding
            stmt = (
                select(
                    TableEmbedding.table_id.label("datasource_embedding_id"),
                    Datasource.description.label("datasource_name"),
                    (1 - TableEmbedding.embedding.cosine_distance(query_embedding)).label("similarity_score"),
                    Datasource.id.label("datasource_id")
                )
                .join(Datasource, TableEmbedding.table_id == Datasource.id)
                .where(TableEmbedding.table_id.in_(datasource_ids))
                .order_by((1 - TableEmbedding.embedding.cosine_distance(query_embedding)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()

            if results:
                # Extract datasource IDs and log messages for each result
                datasource_ids = [str(row.datasource_id) for row in results]
                datasource_names = [row.datasource_name for row in results]
                similarity_scores = [row.similarity_score for row in results]

                state["datasource_ids"] = datasource_ids
                success_message = AIMessage(
                    content=(
                        f"Found {len(datasource_ids)} relevant datasources:\n" +
                        "\n".join(
                            f"Datasource: {name}, Similarity Score: {score:.2f}"
                            for name, score in zip(datasource_names, similarity_scores)
                        )
                    )
                )
                return {"messages": [success_message]}
            else:
                # No results found
                no_results_message = AIMessage(content="No matching datasources found for the query.")
                state["datasource_ids"] = []
                return {"messages": [no_results_message]}
        else:
            # No datasources found for the given domain IDs
            no_datasources_message = AIMessage(content="No datasources found for the given domain IDs.")
            state["datasource_ids"] = []
            return {"messages": [no_datasources_message]}
        

def column_agent(state: dict, config) -> dict:
    engine = config.get('configurable', {}).get('engine')
    limit = config.get('configurable', {}).get('limit', 5)
    datasource_ids = state.get("datasource_ids", [])
    embedding_vector = state.get("embedding_vector", [])
    messages = state.get("messages", [])

    if not datasource_ids:
        error_message = AIMessage(content="No datasource IDs provided.")
        messages.append(error_message)
        state["status"] = "Error: No Datasource IDs"
        return {"messages": messages}

    if not embedding_vector:
        error_message = AIMessage(content="No embedding vector provided.")
        messages.append(error_message)
        state["status"] = "Error: No Embedding Vector"
        return {"messages": messages}
    with Session(engine) as session:
        # Step 1: Filter Column IDs associated with the given datasource IDs
        column_ids_query = (
            select(DataColumn.id, DataColumn.column_dec)
            .where(DataColumn.datasource_id.in_(datasource_ids))
        )
        column_results = session.execute(column_ids_query).all()
        column_ids = [row[0] for row in column_results]

        # Step 2: Perform similarity search on ColumnEmbedding
        if column_ids:
            stmt = (
                select(
                    ColumnEmbedding.column_id,
                    DataColumn.column_dec.label("column_description"),
                    (1 - ColumnEmbedding.embedding.cosine_distance(embedding_vector)).label("similarity_score")
                )
                .join(DataColumn, ColumnEmbedding.column_id == DataColumn.id)
                .where(ColumnEmbedding.column_id.in_(column_ids))
                .order_by((1 - ColumnEmbedding.embedding.cosine_distance(embedding_vector)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()

            # Step 3: Return results
            selected_columns = [
                (row.column_id, row.column_description, row.similarity_score) for row in results
            ]
            state["selected_columns"] = selected_columns

            # Log selected columns
            column_details = "; ".join(
                f"{col_id}: {desc} (Score: {score:.2f})"
                for col_id, desc, score in selected_columns
            )
            success_message = AIMessage(content=f"Selected columns: {column_details}")
            messages.append(success_message)

        else:
            # No matching columns
            state["selected_columns"] = []
            warning_message = AIMessage(content="No matching columns found for the given datasource IDs.")
            messages.append(warning_message)
            state["status"] = "Warning: No Columns Found"

    return {"messages": messages}



workflow = StateGraph(
    state_schema=Dict[str, Any], 
    config_schema=GraphConfig
)



workflow.add_node("Intent Agent", intent_agent)
workflow.add_node("Column Agent",column_agent)
workflow.add_node("Domain Agent",domain_agent)
workflow.add_node("Datasource Agent",datasources_agent)

workflow.add_edge(START,"Intent Agent")
workflow.add_edge("Intent Agent","Domain Agent")
# workflow.add_edge("Intent Agent",END)

workflow.add_edge("Domain Agent","Datasource Agent")
workflow.add_edge("Datasource Agent","Column Agent")
workflow.add_edge("Column Agent",END)


# Compile the workflow
app = workflow.compile()




def extract_final_output_1(app, query: str, engine: str, limit: int = 10) -> Dict[str, Optional[State]]:
    """
    Extracts the final output and state from the LangGraph app during streaming execution.

    Args:
        app: The LangGraph application instance.
        query (str): The user prompt to execute.
        engine (str): The engine to use for execution.
        limit (int): The limit for processing.

    Returns:
        Dict[str, Optional[State]]: A dictionary containing the final output, full state, and any error information.
    """
    final_output = None
    full_state: Optional[State] = None

    try:
        for event in app.stream(
            {"user_prompt": query},  # Adjusted to pass user_prompt directly
            config={
                "engine": engine,
                "limit": limit
            }
        ):
            # Ensure event is a dictionary to match State structure
            if isinstance(event, dict):
                full_state = event  # Capture the current state

                # Extract the final output from the messages
                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        final_output = last_message.content
                    else:
                        final_output = str(last_message)

        return {
            "final_output": final_output,
            "full_state": full_state
        }

    except Exception as e:
        print(f"Error during streaming: {e}")
        return {
            "final_output": None,
            "full_state": None,
            "error": str(e)
        }

# Usage Example
# if __name__ == "__main__":
    # Mock app and engine for demonstration purposes

    class MockApp:
        def stream(self, input_state, config):
            # Mock streaming logic
            yield {
                "messages": [{"content": "Mock response"}],
                "user_prompt": input_state.get("user_prompt"),
                "config": config
            }

    mock_app = MockApp()
    query = "Find the most relevant domains for analyzing sales data trends."

    result = extract_final_output(mock_app, query, engine)
    final_output = result.get("final_output")
    full_state = result.get("full_state")

    print("Final Output:", final_output)
    print("Full State:", full_state)


def extract_final_output(app, query: str, engine: str, limit: int = 10) -> Dict[str, Optional[State]]:
    final_output = None
    full_state: Optional[State] = None

    try:
        for event in app.stream(
            {"user_prompt": query},
            config={
                "engine": engine,
                "limit": limit
            }
        ):
            if isinstance(event, dict):
                full_state = event

                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if "content" in last_message:
                        final_output = last_message["content"]
                    else:
                        final_output = str(last_message)

        return {
            "final_output": final_output,
            "full_state": full_state
        }

    except Exception as e:
        print(f"Error during streaming: {e}")
        return {
            "final_output": None,
            "full_state": None,
            "error": str(e)
        }


result=extract_final_output(app,"How much did I make last month for each store?",engine)
print(result)
# from langchain_core.runnables.graph import MermaidDrawMethod
# from io import BytesIO
# import matplotlib.pyplot as plt

# def display_mermaid_graph_in_vscode(app):
#   """
#   Displays the Mermaid graph of the given app in VS Code.

#   Args:
#     app: The LangChain app object.

#   Note: 
#     - This function relies on matplotlib to display the image within the VS Code environment.
#     - This approach might not be as interactive as using IPython.display.
#   """
#   mermaid_png = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API) 
  
#   # Convert bytes to image object
#   image = plt.imread(BytesIO(mermaid_png))

#   # Display the image using matplotlib
#   plt.imshow(image)
#   plt.axis('off')  # Remove axis for cleaner visualization
#   plt.show() 

# display_mermaid_graph_in_vscode(app)
# # print(p)