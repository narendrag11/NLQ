from fastapi import FastAPI, Query, Depends
from typing import Optional, List
import json
from uuid import UUID

app = FastAPI()

# Load mock data from JSON
def load_mock_data():
    with open("mock_data.json", "r") as file:
        return json.load(file)

@app.get("/workspace/{workspace_id}/connection/all/")
def fetch_all_connections_api(
    workspace_id: UUID,
    search_text: Optional[str] = Query(None),
    db_type: Optional[str] = Query(None),
    skip: int = Query(1),
    limit: int = Query(6)
):
    results = load_mock_data()

    # Filter based on search_text (if provided)
    if search_text:
        results = [conn for conn in results if search_text.lower() in conn["name"].lower()]

    # Pagination logic
    paginated_results = results[skip - 1 : skip - 1 + limit]

    return {
        "status_code": 200,
        "message": "Domain created successfully",
        "data": paginated_results,
        "num_results": len(results)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)