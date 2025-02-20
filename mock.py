from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Literal

app = FastAPI()

# ----------- Request and Response Models -----------
class QueryRequest(BaseModel):
    workspaceId: str
    domainsFlag: Literal["all", "multiple"]
    threadId: str
    userQuery: str
    responseType: Literal["sql", "data", "summary", "all"]

class QueryResponse(BaseModel):
    threadId: str
    responseType: str
    sqlQuery: Optional[str] = None
    data: Optional[list] = None
    summary: Optional[str] = None


# ----------- Mock Endpoint -----------
@app.post("/mock-query-endpoint", response_model=QueryResponse)
def handle_query(payload: QueryRequest = Body(...)):
    """
    A mock endpoint that demonstrates how to handle:
      - workspaceId
      - domainsFlag ("all" or "multiple")
      - threadId
      - userQuery
      - responseType ("sql", "data", "summary", "all")

    Returns a JSON response with:
      - threadId
      - responseType
      - Optionally sqlQuery, data, summary
    """

    # Just echo back the requested threadId
    thread_id = payload.threadId
    
    # The user can request one or more types of information
    # In this mock, we generate placeholders for each type
    mock_sql = "SELECT * FROM sample_table LIMIT 10;"
    mock_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
    mock_summary = "This is a mock summary of your data."

    # Prepare a minimal response
    response = QueryResponse(
        threadId=thread_id,
        responseType=payload.responseType
    )

    # Conditionally include fields based on the responseType
    if payload.responseType in ["sql", "all"]:
        response.sqlQuery = mock_sql

    if payload.responseType in ["data", "all"]:
        response.data = mock_data

    if payload.responseType in ["summary", "all"]:
        response.summary = mock_summary

    return response