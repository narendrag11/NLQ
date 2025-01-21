from uuid import UUID
from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

#gemini 
import google.generativeai as genai

genai.configure(api_key="AIzaSyCSnW71acAMxW29uPX6N2EeWjnej2BG2TI")
model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)

generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )



# System message
sys_msg = SystemMessage(content="""You are an AI that translates natural language into accurate SQL queries.
Understand user requests, generate correct SQL (handling joins, subqueries, aggregations, etc.), support various SQL dialects, and provide explanations.
Focus on accuracy, clarity, efficiency, and user-friendliness.""")
def one_shot():
    pass
    # Example:
    # User: 'Find names and salaries of employees in 'Sales' earning over $50,000.'
    # You:

    # SQL: SELECT name, salary FROM employees WHERE department = 'Sales' AND salary > 50000;
    # Explanation: This query selects 'name' and 'salary' from 'employees' where 'department' is 'Sales' and 'salary' exceeds $50,000.)


# # state management 
from langgraph import State

# Define the state class to store intermediate results
class QueryState(State):
    # Intent information
    intent: str = None
    entities: dict = None

    # Domain Agent outputs
    top_domains: list = []  # List of tuples [(domain_id, similarity), ...]

    # RAG SQL Retrieval outputs
    candidate_sql: str = None
    candidate_sql_score: float = None

    # Table Agent outputs
    candidate_tables: list = []  # List of tuples [(table_id, similarity), ...]

    # Column Selection outputs
    selected_columns: dict = {}  # {table_id: [(column_id, similarity), ...]}

    # SQL Generation outputs
    generated_sql: str = None

    # Error or status information
    status: str = "Initialized"  # Tracks overall status of the pipeline




def intent_agent(user_input:str):
    # LLM Call to gemini ai 
    pass
def domain_agent(user_intent:str,workspace_id:UUID):
    # DB Call
    pass
def datasource_agent(user_intent:str,domain_id:List[UUID]):
    # DB Call
    pass
def column_agent(user_intent:str,datasource_id:List[UUID]):
    # DB Call
    pass




#TODO
def sql_sample_rag(user_intent:str,datasoruce_id:List[UUID]):
    pass


llm_with_tools = model.bind_functions([intent_agent])
# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("AutoBI", AutoBI)
builder.add_node("Intent_Agent", ToolNode([intent_agent]))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "AutoBI")
builder.add_conditional_edges(
    "AutoBI",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("Intent_Agent", END)
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
