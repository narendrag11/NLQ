from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os 
import getpass
from typing import Dict, List
# Define the schema
class IntentModel(BaseModel):
    intent: str = Field(description="The identified intent")
    # parameters: Dict[str, str] = Field(
    #     description="A dictionary of parameter names and their extracted values"
    # )
    # missing_parameters: List[str] = Field(
    #     default_factory=list,
    #     description="A list of any missing parameters"
    # )
    prompt: str = Field(
        default="",
        description="Clarification question for missing parameters, or an empty string if none"
    )
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here")

# Create the model instance
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Bind the schema to the model
structured_llm = llm.with_structured_output(IntentModel)

# Invoke the model
structured_output = structured_llm.invoke("What are some benefits of using LangChain?")
print(structured_output)



def intent_agent_1(state: State) -> Dict[str, List[AnyMessage]]:
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict


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

    # Define system instruction separately from user query
    # Define system instruction separately from user query
    system_instruction = (
        " You are an expert in Natural Language Understanding (NLU) and intent recognition. You only have to give me intent of the user,"
        # "Your task is to analyze user input and extract structured information."
    )

    # JSON schema to structure the LLM output
    # json_schema = IntentModel.model_json_schema()

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
        # Invoke the LLM with the defined schema
        # llm_with_schema = llm.with_structured_output(json_schema)
        ai_msg = llm.invoke(messages)
        logger.info(f"AI Response: {ai_msg}")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")


    # Validate LLM response structure
    # if not ai_msg or "args" not in ai_msg[0]:
    #     raise ValueError("LLM did not return the expected structured output.")

    # if "intent" not in ai_msg[0]["args"]:
    #     raise ValueError("Intent not found in LLM response.")

    # # Extract structured intent
    # data_to_embed = ai_msg[0]["args"]
    # data_as_string = json.dumps(data_to_embed)
    # logger.info(f"Data to embed: {data_as_string}")

    # Generate embeddings for the intent
    intent_embedding = embeddings.embed_query(ai_msg)
    logger.info("Generated intent embedding.")

    # Update the state with intent and embedding
    state["intent"] = ai_msg
    state["embedding_vector"] = intent_embedding

    # Create a message reflecting the extracted intent
    intent_message = AIMessage(content=f"Extracted intent: {ai_msg}")
    logger.info(f"Intent extracted: {intent_embedding}")
    return {"messages": [intent_message]}
    # except Exception as e:
    #     logger.error(f"Intent Agent failed: {e}")
    #     state["status"] = "Intent Extraction Failed"
    #     error_message = AIMessage(content="Failed to extract intent. Please try again.")
    #     return {"messages": [error_message]}
