"""
mock_server.py

A single-file FastAPI mock server that:
1. Generates random v4 UUIDs for 3 workspace IDs and 3 db_type IDs.
2. Creates 20 mock "connections", each with a random v4 UUID.
3. Exposes a single endpoint to list/filter/paginate connections by workspace,
   optional db_type, and optional search_text (substring matching).
4. Summarizes total tables in each connection.

Run this file directly: python mock_server.py
"""

# ====================================================
# 1) Imports and Setup
# ====================================================
import uvicorn
import uuid
import random
from typing import List, Dict, Optional
from fastapi import FastAPI, APIRouter, Query, Depends, HTTPException
from uuid import UUID

# ====================================================
# 2) Generate Mock Data
# ====================================================

# -- Generate 3 random workspace UUIDs
WORKSPACE_IDS = [str(uuid.uuid4()) for _ in range(3)]

# -- Generate 3 random DB type UUIDs
DB_TYPES = [str(uuid.uuid4()) for _ in range(3)]

# -- Some sample table patterns (dict schema -> list[tables])
TABLE_PATTERNS = [
    {"public": ["table_a", "table_b"]},
    {"main": ["data1", "data2", "data3"]},
    {"public": ["alpha", "beta", "gamma"], "archive": ["old_alpha", "old_beta"]},
    {"biz": ["customers", "orders"], "analytics": ["reports"]},
    {"public": ["x", "y", "z"], "extra": ["aux"]}
]

# -- Names for the 20 mock connections
NAMES = [
    "SalesDB", "UserService", "InventoryDB", "AnalyticsService", "BillingDB",
    "ReportsDB", "LegacySystem", "AccountsDB", "MonitoringDB", "IdentityService",
    "PaymentSystem", "WarehouseDB", "MarketingDB", "ProfileStore", "CatalogDB",
    "OrdersHub", "CRMSystem", "ECommerceDB", "LoyaltyProgram", "ArchiverDB"
]

# -- db_name values for the 20 mock connections
DB_NAMES = [
    "sales_db", "userservice_db", "inventory_db", "analytics_db", "billing_db",
    "reports_db", "legacy_db", "accounts_db", "monitor_db", "identity_db",
    "payment_db", "warehouse_db", "marketing_db", "profile_db", "catalog_db",
    "ordershub_db", "crm_db", "ecommerce_db", "loyalty_db", "archiver_db"
]

NUM_CONNECTIONS = 20

# -- Build the list of mock connections
MOCK_CONNECTIONS = []
for i in range(NUM_CONNECTIONS):
    # Choose a workspace ID in round-robin fashion
    w_id = WORKSPACE_IDS[i % len(WORKSPACE_IDS)]
    # Generate a random v4 UUID for each connection
    c_id = str(uuid.uuid4())
    # Randomly pick one of the 3 DB types
    c_db_type = random.choice(DB_TYPES)
    # Use the i-th name and db_name from our lists
    c_name = NAMES[i]
    c_db_name = DB_NAMES[i]
    # Choose a random pattern of schemas -> tables
    c_tables = random.choice(TABLE_PATTERNS)

    connection = {
        "id": c_id,
        "workspace_id": w_id,
        "name": c_name,
        "db_name": c_db_name,
        "db_type": c_db_type,
        "tables": c_tables
    }
    MOCK_CONNECTIONS.append(connection)

# ====================================================
# 3) Dependencies
# ====================================================

def get_current_user():
    """
    Mock user dependency: Pretend we've verified a user.
    """
    return {"username": "mock_user"}

def get_db():
    """
    Mock DB session dependency.
    """
    return None

# ====================================================
# 4) Endpoint Logic and Router
# ====================================================
router = APIRouter()

def get_tables_from_connection(_session, connection_id: str) -> Dict[str, List[str]]:
    """
    Returns the dictionary of schema -> tables for a given connection ID.
    """
    for conn in MOCK_CONNECTIONS:
        if conn["id"] == connection_id:
            return conn["tables"]
    return {}

def matches_similarity(connection: dict, search_text: str) -> bool:
    """
    A naive approach: return True if search_text is in
    connection['name'] or connection['db_name'] (case-insensitive).
    """
    text_lower = search_text.lower()
    return (
        text_lower in connection["name"].lower()
        or text_lower in connection["db_name"].lower()
    )

def pagination_handling(items: List[dict], skip: int, limit: int) -> List[dict]:
    """
    Interprets 'skip' as page index (1-based), and 'limit' as page size.
    """
    start_index = (skip - 1) * limit
    end_index = start_index + limit
    return items[start_index:end_index]

def filter_connections(workspace_id: UUID, db_type: Optional[UUID], search_text: Optional[str],
                       skip: int, limit: int) -> (List[dict], int):
    """
    Filters by workspace_id, db_type, and search_text, then
    calculates total_tables_in_db, then paginates.
    Returns (paged_results, total_count).
    """
    # 1) Filter by workspace_id
    filtered = [c for c in MOCK_CONNECTIONS if c["workspace_id"] == str(workspace_id)]

    # 2) Filter by db_type if given
    if db_type:
        filtered = [c for c in filtered if c["db_type"] == str(db_type)]

    # 3) Filter by naive substring search if given
    if search_text:
        filtered = [c for c in filtered if matches_similarity(c, search_text)]

    # 4) For each connection, compute total_tables_in_db
    for conn in filtered:
        table_dict = get_tables_from_connection(None, conn["id"])
        table_count = sum(len(tbls) for tbls in table_dict.values())
        conn["total_tables_in_db"] = table_count

    # 5) Paginate
    total_count = len(filtered)
    paged = pagination_handling(filtered, skip, limit)
    return paged, total_count

@router.get("/workspace/{workspace_id}/connection/all")
def fetch_all_connections_api(
    workspace_id: UUID,
    search_text: Optional[str] = Query(None),
    db_type: Optional[UUID] = Query(None),
    skip: int = Query(1, description="Page index (starting at 1)"),
    limit: int = Query(6, description="Number of items per page"),
    user=Depends(get_current_user),
    session=Depends(get_db)
):
    """
    Endpoint to fetch all connections for a given workspace.
    - Optionally filter by db_type (UUID).
    - Optionally filter by substring in name or db_name (search_text).
    - Summarize total tables in each connection.
    - Paginate using skip (page) and limit (page size).
    """
    connections, total_count = filter_connections(workspace_id, db_type, search_text, skip, limit)
    return {
        "status_code": 200,
        "message": "Connections fetched successfully",
        "data": connections,
        "num_results": total_count
    }

# ====================================================
# 5) Create FastAPI App and Include Router
# ====================================================
app = FastAPI(title="Mock Connections API (Single File)", version="0.1.0")
app.include_router(router)

# ====================================================
# 6) Run Uvicorn (if invoked directly)
# ====================================================
if __name__ == "__main__":
    uvicorn.run("mock_server:app", host="0.0.0.0", port=8000, reload=True)