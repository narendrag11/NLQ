import os
from google import genai
from langchain.vectorstores import FAISS
from langchain.schema import Document
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import getpass
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class Entities(BaseModel):
    """
    Structured entities extracted from the user's natural language query.
    """
    subject: Optional[str] = Field(
        None, description="The main subject or focus of the query, e.g., 'weather', 'flights', 'restaurants'."
    )
    location: Optional[str] = Field(
        None, description="The location mentioned in the query, e.g., 'New York', 'Paris'."
    )
    datetime: Optional[str] = Field(
        None, description="A date or time specified in the query, e.g., 'tomorrow', 'December 20th'."
    )
    numeric_value: Optional[str] = Field(
        None, description="A numeric value mentioned in the query, e.g., '10', '500'."
    )

class IntentModel(BaseModel):
    """
    Represents the intent and entities extracted from a natural language query.
    """
    intent: str = Field(
        description="The identified intent, such as 'GetWeather', 'BookFlight', 'FindRestaurant'."
    )
    entities: Entities = Field(
        default_factory=Entities,
        description="Structured information about extracted entities like subject, location, and datetime."
    )
    prompt: str = Field(
        default="",
        description="Clarification question for missing or ambiguous parameters, or an empty string if none."
    )

json_schema = IntentModel.model_json_schema()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def intent_agent(user_query:str):
    
    system_instruction= """You are an expert in Natural Language Understanding (NLU) and intent recognition. 
    Your task is to analyze user input and extract structured information.
    """ + user_query 

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
        ).with_structured_output(json_schema)
    
    messages = [
    {"role": "system", "content": system_instruction},  # System message
    {"role": "user", "content": user_query},  # User query
        ]

    ai_msg = llm.invoke(messages)
    data_to_embed = ai_msg[0]["args"]
    data_as_string = json.dumps(data_to_embed)
    intent_embedding = embeddings.embed_query(data_as_string)
    return ai_msg,intent_embedding 



user_input = "How much did I make last month for each store?"
intent_embedded = intent_agent(user_input)
print(intent_embedded)
# data_to_embed = extracted_data[0]["args"]


# data_as_string = json.dumps(data_to_embed)

# intent_embedding = embeddings.embed_query(data_as_string)

# print(intent_embedding)
