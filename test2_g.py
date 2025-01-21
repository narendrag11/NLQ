import json
import logging
from typing import Annotated, List, Dict, Any, Optional, Tuple
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
from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import Session
from models import *  # Ensure your models are correctly defined in this module
from dotenv import load_dotenv
import os

# ------------------------ Configuration and Setup ------------------------

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Google API key from environment variables or prompt the user
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here: ")
    google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure logging to display information and debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your PostgreSQL database URL
DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"  # Update with your actual credentials

# Initialize the SQLAlchemy Database Engine
engine = create_engine(DATABASE_URL)

# Test database connection
try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    logger.info("Database connection successful.")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise

# Initialize embeddings and language model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ------------------------ TypedDict Definitions ------------------------

# class GraphConfig(TypedDict):
#     engine: str
#     limit: int

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_prompt: Optional[str]
    intent: Optional[Dict[str, Any]]
    embedding_vector: Optional[List[float]]
    domain_ids: Optional[List[str]]
    datasource_ids: Optional[List[UUID]]
    selected_columns: Optional[List[Tuple[UUID, str, float]]]
    sql_query: Optional[str]
    sql_queries: Optional[List[str]]
    fallback: Optional[bool]
    status: Optional[str]
    # config: GraphConfig
    # engine:Optional[str]=create_engine("postgresql://postgres:password@localhost:5432/Autobi_DB")
    # TODO: Add user_id, workspace_id, etc., as needed

# ------------------------ Agent Definitions ------------------------

def intent_agent_2(state: State) -> Dict[str, List[AnyMessage]]:
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict
    import json
    import logging

    logger = logging.getLogger(__name__)

    class Filters(BaseModel):
        """
        Represents additional filters or constraints extracted from the user input.
        """
        filters: Optional[Dict[str, str]] = Field(
            default_factory=dict,
            description="Additional filters or constraints extracted from the input, e.g., {'date': '2025-01-09'}."
        )

    class IntentModel(BaseModel):
        intent: str = Field(
            default=None,
            description="The identified intent, such as 'GetWeather', 'BookFlight', or 'FindRestaurant'."
        )
        entities: List[str] = Field(
            default_factory=list,
            description="A list of entities extracted from the query, e.g., ['New York', 'tomorrow']."
        )
        filters: Optional[Filters] = Field(
            default=None,
            description="Additional filters or structured constraints derived from the input."
        )

    logger.info("Starting Intent Agent")
    user_query = state.get("user_prompt", "")
    user_query="What is the average transaction amount and the most common discount percentage applied to sales transactions?"
    if not user_query:
        state["status"] = "No User Prompt"
        logger.warning("No user prompt provided.")
        return {"messages": [AIMessage(content="No user prompt provided.")]}

    # Define system instruction
    system_instruction = (
        "You are an expert in Natural Language Understanding (NLU) and intent recognition. "
        "Your task is to analyze user input and extract the intent."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_retries=2,  # Retry in case of transient errors
    )

    # Prepare LLM messages
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query},
    ]

    logger.info(f"LLM object: {llm}")
    try:
        # Invoke the LLM
        ai_msg = llm.invoke(messages)
        logger.info(f"AI Response: {ai_msg}")

        # Validate LLM response structure
        if not ai_msg or "args" not in ai_msg[0]:
            raise ValueError("LLM did not return the expected structured output.")

        if "intent" not in ai_msg[0]["args"]:
            raise ValueError("Intent not found in LLM response.")

        # Extract structured intent
        data_to_embed = ai_msg[0]["args"]
        data_as_string = json.dumps(data_to_embed)
        logger.info(f"Data to embed: {data_as_string}")

        # Generate embeddings for the intent
        intent_embedding = embeddings.embed_query(data_as_string)
        logger.info("Generated intent embedding.")

        # Update the state with intent and embedding
        state["intent"] = data_to_embed
        state["embedding_vector"] = intent_embedding

        # Create a message reflecting the extracted intent
        intent_message = AIMessage(content=f"Extracted intent: {data_to_embed}")
        logger.info(f"Intent extracted: {data_to_embed}")
        return {"messages": [intent_message]}

    except Exception as e:
        logger.error(f"Intent Agent failed: {e}")
        state["status"] = "Intent Extraction Failed"
        error_message = AIMessage(content="Failed to extract intent. Please try again.")
        return {"messages": [error_message]}

def intent_agent(state: State) -> Dict[str, List[AnyMessage]]:
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict
    import json
    import logging

    logger = logging.getLogger(__name__)

    class Filters(BaseModel):
        """
        Represents additional filters or constraints extracted from the user input.
        """
        filters: Optional[Dict[str, str]] = Field(
            default_factory=dict,
            description="Additional filters or constraints extracted from the input, e.g., {'date': '2025-01-09'}."
        )

    class IntentModel(BaseModel):
        intent: str = Field(
            default=None,
            description="The identified intent, such as 'GetWeather', 'BookFlight', or 'FindRestaurant'."
        )
        entities: List[str] = Field(
            default_factory=list,
            description="A list of entities extracted from the query, e.g., ['New York', 'tomorrow']."
        )
        filters: Optional[Filters] = Field(
            default=None,
            description="Additional filters or structured constraints derived from the input."
        )

    logger.info("Starting Intent Agent")
    user_query = state.get("user_prompt", "")
    # user_query="What is the average transaction amount and the most common discount percentage applied to sales transactions?"
    if not user_query:
        state["status"] = "No User Prompt"
        logger.warning("No user prompt provided.")
        return {"messages": [AIMessage(content="No user prompt provided.")]}

    # Define system instruction
    system_instruction = (
        "You are an expert in Natural Language Understanding (NLU) and intent recognition. "
        "Your task is to analyze user input and extract the intent."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_retries=2,  # Retry in case of transient errors
    )

    # Prepare LLM messages
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query},
    ]

    logger.info(f"LLM object: {llm}")
    try:
        # Invoke the LLM
        ai_msg = llm.invoke(messages)
        logger.info(f"AI Response: {ai_msg}")

        # Validate LLM response structure
        # if not ai_msg or not isinstance(ai_msg, list) or len(ai_msg) == 0:
        #     raise ValueError("LLM did not return the expected structured output.")

        ai_content = ai_msg.content  # Access the content of the AIMessage
        # if not isinstance(ai_content, dict) or "intent" not in ai_content:
        #     raise ValueError("Intent not found in LLM response.")
        logger.info(f"Ai Content is {ai_content}")
        logger.info(f"type is {type(ai_content)}")
        # Extract structured intent
        data_to_embed = ai_content
        # data_as_string = json.dumps(data_to_embed)
        logger.info(f"Data to embed: {data_to_embed}")

        # Generate embeddings for the intent
        intent_embedding = embeddings.embed_query(data_to_embed)
        logger.info("Generated intent embedding.")

        # Update the state with intent and embedding
        state["intent"] = data_to_embed
        state["embedding_vector"] = intent_embedding
        logger.info(f"embedding vector {state["embedding_vector"]}")
        # Create a message reflecting the extracted intent
        intent_message = AIMessage(content=f"Extracted intent: {data_to_embed}")
        logger.info(f"Intent extracted: {data_to_embed}")
        return {"embedding_vector": intent_embedding}

    except Exception as e:
        logger.error(f"Intent Agent failed: {e}")
        state["status"] = "Intent Extraction Failed"
        error_message = AIMessage(content="Failed to extract intent. Please try again.")
        return {"messages": [error_message]}

def domain_agent(state: State) -> Dict[str, List[AnyMessage]]:
    logger.info("Starting Domain Agent")
    engine = create_engine("postgresql://postgres:password@localhost:5432/Autobi_DB")
    limit = 2
    logger.info(engine)
    query_intent_embedding = state.get("embedding_vector",[])
    if not isinstance(query_intent_embedding, list):
        print(f"Not valid --- {query_intent_embedding}")
    # logger.info(f"embedding12 {query_intent_embedding}")
    # logger.info(f"messgae from state is  {state.get("messages")}")
    if not query_intent_embedding:
        logger.error("No embedding vector found in state.")
        state["status"] = "Missing Embedding Vector"
        return {"messages": [AIMessage(content="Missing embedding vector.")]}

    try:
        with Session(engine) as session:
            stmt = (
                select(
                    DomainEmbedding,  
                    Domain.id.label("domain_id"),    
                    Domain.name.label("domain_name"),  
                    (1 - DomainEmbedding.embedding.cosine_distance(query_intent_embedding)).label("similarity_score")
                )
                .join(Domain, DomainEmbedding.domain_id == Domain.id)
                .order_by((1 - DomainEmbedding.embedding.cosine_distance(query_intent_embedding)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()
            
            # Modify unpacking to match actual result structure
            domains = []
            for result in results:
                # Unpack carefully
                if len(result) == 4:  # Ensure 4 elements
                    domain_embedding, domain_id, domain_name, similarity_score = result
                    domains.append({
                        "domain_id": str(domain_id),
                        "domain_name": domain_name,
                        "similarity_score": round(similarity_score, 4),
                    })
                else:
                    logger.warning(f"Unexpected result structure: {result}")


            if not domains:
                logger.warning("No domains found.")
                state["status"] = "No Domains Found"
                domain_message = AIMessage(content="No relevant domains found.")
                return {"messages": [domain_message]}

            # Update state with retrieved domain IDs
            state["domain_ids"] = [domain["domain_id"] for domain in domains]
            logger.info(f"Retrieved domain IDs: {state['domain_ids']}")

            # Create a success message with domain details
            domain_message = AIMessage(content=json.dumps({"domains": domains}, indent=2))
            logger.info(f"Domain Agent message: {domain_message}")
            return {"domain_ids": state['domain_ids'],"embedding_vector":state["embedding_vector"]}
    except Exception as e:
        logger.error(f"Domain Agent failed: {e}")
        state["status"] = "Domain Retrieval Failed"
        error_message = AIMessage(content="Failed to retrieve domains.")
        return {"messages": [error_message]}

def datasources_agent(state: State) -> Dict[str, List[AnyMessage]]:
    logger.info("Starting Datasources Agent")
    engine=create_engine("postgresql://postgres:password@localhost:5432/Autobi_DB")
    # engine = config.get('engine')
    # limit = config.get('limit', 5)
    limit=4
    domain_ids = state.get("domain_ids", [])
    query_embedding = state.get("embedding_vector", [])

    # Ensure required parameters are present
    logger.info(f"datasource --- domain_id {domain_ids}")
    # logger.info(f"datasource --- embedding {query_embedding}")

    if not domain_ids or not query_embedding or engine is None:
        error_message = AIMessage(content="Missing required parameters: domain_ids, query_embedding, or engine.")
        logger.error("Missing required parameters in Datasources Agent.")
        state["status"] = "Error"
        return {"messages": [error_message]}

    try:
        with Session(engine) as session:
            # Step 1: Retrieve Datasource IDs associated with the given Domain IDs
            datasource_ids_query = (
                select(DomainDatasource.datasource_id)
                .where(DomainDatasource.domain_id.in_(domain_ids))
            )
            datasource_ids = [row[0] for row in session.execute(datasource_ids_query).all()]
            logger.info(f"Retrieved datasource IDs: {datasource_ids}")

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
                logger.info(f"Datasource similarity search results: {results}")

                if results:
                    # Extract and log datasource details
                    datasource_ids_result = [str(row.datasource_id) for row in results]
                    datasource_names = [row.datasource_name for row in results]
                    similarity_scores = [row.similarity_score for row in results]

                    state["datasource_ids"] = datasource_ids_result
                    logger.info(f"Set datasource_ids in state: {state['datasource_ids']}")

                    success_message = AIMessage(
                        content=(
                            f"Found {len(datasource_ids_result)} relevant datasources:\n" +
                            "\n".join(
                                f"Datasource: {name}, Similarity Score: {score:.2f}"
                                for name, score in zip(datasource_names, similarity_scores)
                            )
                        )
                    )
                    logger.info(f"Datasources Agent success message: {success_message}")
                    return {"datasource_ids": datasource_ids_result,"embedding_vector":query_embedding}
                else:
                    # No matching datasources found
                    no_results_message = AIMessage(content="No matching datasources found for the query.")
                    state["datasource_ids"] = []
                    logger.warning("No matching datasources found.")
                    return {"datasource_ids": [no_results_message]}
            else:
                # No datasources associated with the given domain IDs
                no_datasources_message = AIMessage(content="No datasources found for the given domain IDs.")
                state["datasource_ids"] = []
                logger.warning("No datasources found for the given domain IDs.")
                return {"messages": [no_datasources_message]}
    except Exception as e:
        logger.error(f"Datasources Agent failed: {e}")
        state["status"] = "Datasources Retrieval Failed"
        error_message = AIMessage(content="Failed to retrieve datasources.")
        return {"messages": [error_message]}

def column_agent(state: State) -> Dict[str, List[AnyMessage]]:
    logger.info("Starting Column Agent")
    engine = create_engine("postgresql://postgres:password@localhost:5432/Autobi_DB")
    limit = 5
    datasource_ids = state.get("datasource_ids", [])
    embedding_vector = state.get("embedding_vector", [])
    messages = state.get("messages", [])

    # Validate presence of datasource IDs and embedding vector
    if not datasource_ids:
        error_message = AIMessage(content="No datasource IDs provided.")
        logger.error("No datasource IDs provided.")
        messages.append(error_message)
        state["status"] = "Error: No Datasource IDs"
        return {"messages": messages}

    if not embedding_vector:
        error_message = AIMessage(content="No embedding vector provided.")
        logger.error("No embedding vector provided.")
        messages.append(error_message)
        state["status"] = "Error: No Embedding Vector"
        return {"messages": messages}

    try:
        with Session(engine) as session:
            # Step 1: Retrieve Column IDs associated with the given datasource IDs
            column_ids_query = (
                select(DataColumn.id, DataColumn.column_dec)
                .where(DataColumn.datasource_id.in_(datasource_ids))
            )
            column_results = session.execute(column_ids_query).all()
            column_ids = [row[0] for row in column_results]
            logger.info(f"Retrieved column IDs: {column_ids}")

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
                logger.info(f"Column similarity search results: {results}")

                if results:
                    # Extract and log selected columns
                    selected_columns = [
                        (row.column_id, row.column_description, row.similarity_score) for row in results
                    ]
                    state["selected_columns"] = selected_columns
                    logger.info(f"Selected columns: {selected_columns}")

                    # Create a success message with column details
                    column_details = "; ".join(
                        f"{col_id}: {desc} (Score: {score:.2f})"
                        for col_id, desc, score in selected_columns
                    )
                    success_message = AIMessage(content=f"Selected columns: {column_details}")
                    messages.append(success_message)
                    logger.info(f"Column Agent success message: {success_message}")
                else:
                    # No matching columns found
                    state["selected_columns"] = []
                    warning_message = AIMessage(content="No matching columns found for the given datasource IDs.")
                    messages.append(warning_message)
                    state["status"] = "Warning: No Columns Found"
                    logger.warning("No matching columns found.")
            else:
                # No columns associated with the given datasource IDs
                state["selected_columns"] = []
                warning_message = AIMessage(content="No matching columns found for the given datasource IDs.")
                messages.append(warning_message)
                state["status"] = "Warning: No Columns Found"
                logger.warning("No matching columns found.")

    except Exception as e:
        logger.error(f"Column Agent failed: {e}")
        state["status"] = "Column Retrieval Failed"
        warning_message = AIMessage(content="Failed to retrieve columns.")
        messages.append(warning_message)

    return {"datasource_id": messages}

# ------------------------ Workflow Definition ------------------------

# Initialize the StateGraph with appropriate schemas
workflow = StateGraph(State)

    # config_schema=GraphConfig


# Add agents as nodes in the workflow
workflow.add_node("Intent Agent", intent_agent)
workflow.add_node("Domain Agent", domain_agent)
workflow.add_node("Datasource Agent", datasources_agent)
workflow.add_node("Column Agent", column_agent)

# Define the edges to establish the execution order
workflow.add_edge(START, "Intent Agent")
workflow.add_edge("Intent Agent", "Domain Agent")
workflow.add_edge("Domain Agent", "Datasource Agent")
workflow.add_edge("Datasource Agent", "Column Agent")
workflow.add_edge("Column Agent", END)

# Compile the workflow into an executable application
app = workflow.compile()

# ------------------------ Helper Function ------------------------

def extract_final_output(app) -> Dict[str, Optional[Any]]:
    
    final_output = None
    full_state: Optional[Dict[str, Any]] = None

    try:
        for event in  app.invoke({State["user_prompt"]:query}
    # {"user_prompt": query, "config":{
    #             "engine": engine,
    #             "limit": limit
    #         }}

        ):
            if isinstance(event, dict):
                full_state = event

                messages = event.get("embedding_vector", [])
                # if messages:
                #     last_message = messages[-1]
                    # if "content" in last_message:
                    #     final_output = last_message["content"]
                    # else:
                    #     final_output = str(last_message)

        return {
            "final_output": messages,
            # "full_state": full_state
        }

    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        return {
            "final_output": None,
            "full_state": None,
            "error": str(e)
        }

# ------------------------ Execution Example ------------------------

if __name__ == "__main__":
    # Define the user query
    query = "What is the average transaction amount and the most common discount percentage applied to sales transactions?"

    # Execute the workflow and extract outputs
    result = extract_final_output(app)
    print(result)
    # final_output = result.get("final_output")
    # full_state = result.get("full_state")

