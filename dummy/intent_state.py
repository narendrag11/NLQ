from typing import Annotated, List, Dict, TypedDict
from langgraph.graph.message import add_messages
import requests
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import google.generativeai as genai
genai.configure(api_key="AIzaSyCSnW71acAMxW29uPX6N2EeWjnej2BG2TI")


class State(TypedDict):
    """
    Represents the state in the new LangGraph structure using TypedDict.
    Includes:
        - messages: A log of messages processed during the graph execution.
        - intent: Extracted intent from the user input.
        - entities: Extracted entities (e.g., filters, metrics, etc.).
        - status: Current processing status of the graph.
    """
    messages: Annotated[List[Dict], add_messages]  # List of messages (user and system interactions)
    intent: str
    entities: Dict[str, str]  # Example: {"time_range": "last month", "metric": "sales"}
    status: str


def intent_agent(user_input: str, state: State) -> State:
    """
    Processes user input to extract intent and entities using Gemini LLM,
    and updates the state with the parsed output.

    Args:
        user_input (str): The natural language input from the user.
        state (State): The current state dictionary to be updated.

    Returns:
        State: The updated state dictionary with extracted intent and entities.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = PromptTemplate(
    template="""Extract the user's intent and entities from the following text.
    Provide the output in strict JSON format with the following structure:\n{format_instructions}\n{user_input}\n""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # Prepare the prompt
    prompt = f"""
    Extract the user's intent and entities from the following text.
    Provide the output in strict JSON format with the following structure:
    {{
      "intent": "",
      "entities": {{
        "time_range": "",
        "metric": ""
      }}
    }}

    User Input: {user_input}
    """

    try:
        # Generate response using the Gemini API
        response = model.generate_content(prompt)
        raw_output = response.text.strip()  # Extract the generated content

        print("Raw LLM Output:", raw_output)  # Debugging step to inspect the raw output

        # Attempt to parse the LLM output
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            # Attempt to sanitize and re-parse
            sanitized_output = raw_output.replace("\n", "").strip()
            parsed_output = json.loads(sanitized_output)

        # Validate parsed output
        if "intent" in parsed_output and "entities" in parsed_output:
            return {
                "intent": parsed_output["intent"],
                "entities": parsed_output["entities"],
                "status": "Intent Extracted"
            }
        else:
            return {
                "intent": None,
                "entities": None,
                "status": "Ambiguous Query",
                "error": "Incomplete response from LLM."
            }

    except json.JSONDecodeError as e:
        # Handle parsing errors
        return {
            "intent": None,
            "entities": None,
            "status": "Parsing Error",
            "error": f"Failed to parse sanitized LLM response: {e}"
        }

    except Exception as e:
        # Handle general errors
        return {
            "intent": None,
            "entities": None,
            "status": "Error",
            "error": f"Error calling Gemini API: {e}"
        }
# Initialize state
state: State = {
    "messages": [],
    "intent": None,
    "entities": {},
    "status": "Initialized"
}

# User input
user_input = "What were the total sales for the last month?"

# Call intent_agent
updated_state = intent_agent(user_input, state)

# Output the updated state
print(json.dumps(updated_state, indent=2))
