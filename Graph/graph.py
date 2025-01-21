from typing import Annotated, Literal,Optional,Dict,Any,List,Tuple
from uuid import UUID

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from column_Agent import column_agent
from datasource_agent import datasources_agent
from domain_agent import domain_agent
from intent_agents import intent_agent

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_prompt: Optional[str]  # Capture the initial input
    intent: Optional[Dict[str, Any]]
    embedding_vector: Optional[List[float]]
    top_domains: Optional[List[str]]
    datasource_ids: Optional[List[UUID]]
    selected_columns: Optional[List[Tuple[UUID, str, float]]]
    sql_query: Optional[str]
    sql_queries: Optional[List[str]]
    fallback: Optional[bool]
    status: Optional[str]

# Define a new graph
workflow = StateGraph(State)

