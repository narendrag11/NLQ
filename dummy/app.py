# app.py

from typing import Any, Dict, List, Tuple, Optional
from langgraph import Graph, Node  # Make sure you've installed the latest version
from sqlmodel import create_engine, Session, select
from uuid import UUID
import json

# Import your models
from models import (
    Domain, 
    Datasource, 
    DataColumn, 
    DomainEmbedding,
    TableEmbedding,
    ColumnEmbedding
)

###############################################################################
# 1. Mocked Gemini LLM & Embeddings
###############################################################################
def gemini_llm(prompt: str) -> str:
    """
    Mock function to call Gemini LLM with a prompt.
    Returns a JSON-like string for illustration.
    
    """
    # For demonstration, let's return a fake JSON response
    # that the "IntentAgentNode" might produce.
    # e.g. extracting user intent with filters, etc.
    mock_response = {
        "intent": "Get monthly sales",
        "entities": {
            "time_range": "last month",
            "metric": "sales"
        }
    }
    return json.dumps(mock_response)


def gemini_embedding(text: str) -> List[float]:
    """
    Mock function to generate an embedding from the Gemini model.
    Replace with the actual embedding logic from your LLM.
    """
    import random
    # Just returning a random 768-dim vector for demonstration.
    return [random.random() for _ in range(768)]


###############################################################################
# 2. Nodes (Agents / Chains) Definitions
###############################################################################

class IntentAgentNode(Node):
    """
    2.1 Intent Agent Node
    Inputs: User’s natural language prompt.
    Outputs: Structured representation of intent (entities, filters, relationships, etc.).
    Implementation:
      - Call Gemini LLM with a prompt template
      - Parse the response into a JSON or dict
    Edge Cases:
      - If ambiguous, return an “Ambiguous Query” status.
    """
    def run(self, user_prompt: str) -> Dict[str, Any]:
        # Call Gemini LLM
        llm_output = gemini_llm(user_prompt)

        try:
            structured_intent = json.loads(llm_output)
        except json.JSONDecodeError:
            # If we cannot parse properly, treat as ambiguous
            return {
                "status": "Ambiguous Query",
                "error": "Could not parse LLM response"
            }

        # Basic check for required fields
        if "intent" not in structured_intent:
            structured_intent["status"] = "Ambiguous Query"
            structured_intent["error"] = "No clear intent found"
        
        return structured_intent


class DomainAgentNode(Node):
    """
    2.2 Domain Agent Node
    Inputs: Extracted intent (keywords, domain hints).
    Outputs: Top 5 relevant domains or fewer if matches are insufficient.
    Implementation:
      - Generate an embedding vector from the intent text
      - Query domain_embeddings in PostgreSQL to find top 5 domains by cosine similarity
    Edge Cases:
      - If top domains have a similarity score below threshold (e.g., 0.4), return “Domain Not Found”
    """
    def __init__(self, session_factory, threshold=0.4, **kwargs):
        super().__init__(**kwargs)
        self.session_factory = session_factory
        self.threshold = threshold

    def run(self, structured_intent: Dict[str, Any]) -> Dict[str, Any]:
        # If there's an error or no intent, propagate
        if structured_intent.get("status") == "Ambiguous Query":
            return structured_intent

        intent_text = structured_intent.get("intent", "")
        if not intent_text:
            return {
                "status": "Domain Not Found",
                "error": "No intent text provided"
            }

        # Generate embedding
        user_emb = gemini_embedding(intent_text)

        with self.session_factory() as session:
            # Example using pgvector <-> operator for similarity
            sql_query = """
                SELECT 
                    domain_id, 
                    embedding <-> CAST(:user_emb as vector) AS similarity
                FROM embedding.domainembedding
                ORDER BY similarity ASC
                LIMIT 5
            """
            results = session.exec(sql_query, {"user_emb": user_emb}).all()

        top_domains = []
        for row in results:
            domain_id, similarity = row
            if similarity > (1 - self.threshold):
                # Because lower <-> means more similar, we do 1 - threshold check as an example
                continue
            top_domains.append((domain_id, similarity))

        if not top_domains:
            structured_intent["status"] = "Domain Not Found"
            structured_intent["error"] = "No domain match above similarity threshold."
            structured_intent["top_domains"] = []
        else:
            structured_intent["top_domains"] = top_domains  # list of (domain_id, similarity)
        
        return structured_intent


class RAGSQLRetrievalNode(Node):
    """
    2.3 RAG SQL Retrieval Node
    Inputs: Top 5 domains, user’s structured intent.
    Outputs: A matching SQL query or a fallback signal.
    Implementation:
      - For each domain, fetch candidate queries from some 'sql_knowledge_base'
      - Compute similarity with user’s intent embedding
      - If match ≥ 0.9, return that query immediately
    Edge Cases:
      - Multiple High-Similarity Matches
      - No Matching Query -> fallback
    """
    def __init__(self, session_factory, match_threshold=0.9, **kwargs):
        super().__init__(**kwargs)
        self.session_factory = session_factory
        self.match_threshold = match_threshold

    def run(self, structured_intent: Dict[str, Any]) -> Dict[str, Any]:
        if structured_intent.get("status") in ["Ambiguous Query", "Domain Not Found"]:
            # Pass along
            return structured_intent

        top_domains = structured_intent.get("top_domains", [])
        if not top_domains:
            structured_intent["status"] = "No Domain Provided"
            return structured_intent

        # In a real system, you might have a table 'sql_knowledge_base' with pre-built queries
        # We'll mock logic with a simple placeholder
        user_emb = gemini_embedding(structured_intent["intent"])  # re-generate or reuse

        best_match_query = None
        best_score = 0.0

        # Mocked example of iterating domain "knowledge base"
        # In reality, you'd retrieve from "sql_knowledge_base" table.
        candidate_queries = [
            {"domain_id": top_domains[0][0], "query_text": "SELECT SUM(sales) FROM sales_table WHERE date > '2021-01-01';"},
            {"domain_id": top_domains[0][0], "query_text": "SELECT COUNT(*) FROM user_table;"}
        ]
        
        for candidate in candidate_queries:
            # Mock some similarity: random for demonstration
            import random
            similarity = random.random()

            if similarity > self.match_threshold and similarity > best_score:
                best_score = similarity
                best_match_query = candidate["query_text"]

        if best_match_query:
            structured_intent["candidate_sql"] = best_match_query
            structured_intent["status"] = "SQL Found"
        else:
            # Indicate we need to fallback to the Table Agent
            structured_intent["status"] = "Fallback"
        
        return structured_intent


class TableAgentNode(Node):
    """
    2.4 Table Agent Node (Fallback)
    Inputs: Structured intent, top 5 domains (or chosen domain).
    Outputs: List of candidate tables sorted by relevance.
    """
    def __init__(self, session_factory, num_tables=5, **kwargs):
        super().__init__(**kwargs)
        self.session_factory = session_factory
        self.num_tables = num_tables

    def run(self, structured_intent: Dict[str, Any]) -> Dict[str, Any]:
        # If we already have a SQL or no need to fallback, pass through
        if structured_intent.get("status") not in ["Fallback"]:
            return structured_intent

        top_domains = structured_intent.get("top_domains", [])
        if not top_domains:
            structured_intent["status"] = "No Relevant Table Found"
            return structured_intent

        user_emb = gemini_embedding(structured_intent["intent"])

        # For each domain, find the relevant tables (datasources)
        candidate_tables = []
        with self.session_factory() as session:
            for domain_id, _sim in top_domains:
                sql_query = """
                    SELECT
                        te.table_id,
                        te.embedding <-> CAST(:user_emb as vector) as similarity
                    FROM embedding.tableembedding te
                    JOIN datasource d ON d.id = te.table_id
                    -- Optionally join domain_datasource dd if needed 
                    -- to filter only datasources that belong to the domain
                    ORDER BY similarity ASC
                    LIMIT :limit;
                """
                results = session.exec(sql_query, {
                    "user_emb": user_emb,
                    "limit": self.num_tables
                }).all()

                # Accumulate tables
                for row in results:
                    table_id, similarity = row
                    candidate_tables.append((table_id, similarity))

        # Sort overall by ascending similarity
        candidate_tables.sort(key=lambda x: x[1])
        structured_intent["candidate_tables"] = candidate_tables

        if not candidate_tables:
            structured_intent["status"] = "No Relevant Table Found"
        else:
            structured_intent["status"] = "Table Candidates Found"

        return structured_intent


class ColumnSelectionAgentNode(Node):
    """
    2.5 Column Selection Agent Node
    Inputs: List of candidate tables, user’s structured intent.
    Outputs: Pruned set of relevant columns for each table.
    """
    def __init__(self, session_factory, max_columns=10, **kwargs):
        super().__init__(**kwargs)
        self.session_factory = session_factory
        self.max_columns = max_columns

    def run(self, structured_intent: Dict[str, Any]) -> Dict[str, Any]:
        status = structured_intent.get("status")
        if status not in ["Table Candidates Found", "Fallback", "No Relevant Table Found"]:
            # Means we either have a direct SQL or no domain
            return structured_intent
        
        # If no tables found, skip
        candidate_tables = structured_intent.get("candidate_tables", [])
        if not candidate_tables:
            return structured_intent  # pass it on

        user_emb = gemini_embedding(structured_intent["intent"])

        final_table_columns = {}  # table_id -> list of (column_id, similarity)
        with self.session_factory() as session:
            for (table_id, _sim) in candidate_tables:
                # Query column embeddings
                col_query = """
                    SELECT 
                        ce.column_id,
                        ce.embedding <-> CAST(:user_emb as vector) AS similarity
                    FROM embedding.columnembedding ce
                    WHERE ce.column_id IN (
                        SELECT id FROM data_column WHERE datasource_id = :table_id
                    )
                    ORDER BY similarity ASC
                """
                col_results = session.exec(col_query, {
                    "user_emb": user_emb,
                    "table_id": table_id
                }).all()

                # Take top N relevant columns
                top_cols = col_results[:self.max_columns]
                final_table_columns[table_id] = top_cols

        structured_intent["selected_columns"] = final_table_columns
        structured_intent["status"] = "Columns Selected"
        return structured_intent


class SQLGenerationNode(Node):
    """
    2.6 SQL Generation Node
    Inputs: Selected tables, columns, user’s intent, metadata (filters, aggregations).
    Outputs: A complete SQL query string.
    """
    def run(self, structured_intent: Dict[str, Any]) -> Dict[str, Any]:
        # If we already found a candidate SQL, let's skip generation
        if structured_intent.get("status") == "SQL Found":
            return structured_intent

        if structured_intent.get("status") != "Columns Selected":
            # No columns or no fallback path
            return structured_intent

        # Let's pretend we want to generate a single SQL query out of the top table & columns
        selected_cols = structured_intent.get("selected_columns", {})
        if not selected_cols:
            structured_intent["status"] = "No SQL Generated"
            return structured_intent

        # For demonstration, pick the first table
        first_table_id = list(selected_cols.keys())[0]
        columns = selected_cols[first_table_id]
        # columns is a list of (column_id, similarity)
        # In a real scenario, we would fetch actual column names from DataColumn

        # Mock getting column names:
        column_names = [f"col_{col_id}" for (col_id, _) in columns]
        if not column_names:
            structured_intent["status"] = "No SQL Generated"
            return structured_intent

        # A very naive SQL generation example
        # In practice, you'd call Gemini or your LLM with a more advanced prompt
        table_name = f"datasource_{first_table_id}"  # or fetch the actual table name from DB
        col_str = ", ".join(column_names)
        where_clause = ""  # you might derive filters from structured_intent
        group_by = ""      # from structured_intent if aggregation is requested

        sql_query = f"SELECT {col_str} FROM {table_name} {where_clause} {group_by};"
        structured_intent["generated_sql"] = sql_query
        structured_intent["status"] = "SQL Generated"

        return structured_intent


###############################################################################
# 3. Building the DAG (Graph)
###############################################################################
def build_pipeline(db_uri: str):
    """
    Build the LangGraph pipeline (DAG) of nodes.
    """
    engine = create_engine(db_uri, echo=False)

    def session_factory():
        return Session(engine)

    graph = Graph(name="SQL_RAG_Pipeline")

    # Create node instances
    intent_node = IntentAgentNode(name="intent_agent")
    domain_node = DomainAgentNode(name="domain_agent", session_factory=session_factory)
    rag_sql_node = RAGSQLRetrievalNode(name="rag_sql_retrieval", session_factory=session_factory)
    table_node = TableAgentNode(name="table_agent", session_factory=session_factory)
    column_node = ColumnSelectionAgentNode(name="column_selection_agent", session_factory=session_factory)
    sql_gen_node = SQLGenerationNode(name="sql_generation_agent")

    # Add them to the graph
    graph.add_nodes([
        intent_node,
        domain_node,
        rag_sql_node,
        table_node,
        column_node,
        sql_gen_node
    ])

    # Connect them in a DAG
    # Each node takes the output of the previous node as input
    # and returns its own structured dictionary.
    graph.add_edges([
        (intent_node, domain_node),
        (domain_node, rag_sql_node),
        (rag_sql_node, table_node),
        (table_node, column_node),
        (column_node, sql_gen_node)
    ])

    return graph


###############################################################################
# 4. Main Execution Example
###############################################################################
if __name__ == "__main__":
    # Example usage
    DATABASE_URI = "postgresql://postgres:password@localhost:5432/Autobi_DB"
    pipeline = build_pipeline(DATABASE_URI)

    user_input = "Show me the sales for the last month"
    result = pipeline.run(user_input)
    
    print("Pipeline output:")
    print(result)
