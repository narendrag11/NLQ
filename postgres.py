import psycopg2
from psycopg2 import OperationalError, Error

def create_connection(db_name: str, user: str, password: str, host: str, port: int):
    conn = None
    try:
        conn = psycopg2.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
    except OperationalError as e:
        # Handle operational errors such as not being able to connect
        print(f"Error connecting to the database: {e}")
    except Exception as e:
        # Handle any other exceptions
        print(f"Unexpected error while creating connection: {e}")
    return conn


def execute_query(conn, query: str):
    if not query or not query.strip():
        print("Error: The query is empty or None.")
        return None

    records = None
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        # Use fetchall() if you expect multiple rows; fetchone() if you want the first row, etc.
        records = cursor.fetchone()
    except Error as e:
        print(f"Error executing query: {e}")
    finally:
        if cursor:
            cursor.close()
    return records


def sql_execution_output_check(
    db_name: str,
    user: str,
    password: str,
    host: str,
    port: int,
    query: str
) -> bool:
    conn = create_connection(db_name, user, password, host, port)
    if not conn:
        return "Invalid"

    try:
        records = execute_query(conn, query)
        if records and len(records) > 0:
            return True
        else:
            return False
    finally:
        
        conn.close()





import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError

def extract_and_validate_tables(
    sql_query: str, 
    allowed_tables: set, 
    dialect: str = "postgres"
) -> set:
    try:
        parsed = sqlglot.parse_one(sql_query, read=dialect, error_level="EXCEPTION")
    except ParseError as e:
        raise ValueError(f"Invalid SQL syntax for {dialect}: {e}")
    disallowed_expressions = (
        exp.Create,
        exp.Alter,
        exp.Update,
        exp.Insert,
        exp.Delete,
        exp.Drop,
        exp.Merge,
        exp.Grant,
        exp.Set,
    )
    for node in parsed.walk():
        if isinstance(node, disallowed_expressions):
            raise ValueError(
                f"The query must be a SELECT (or UNION of SELECTs). "
                f"Found disallowed statement: {node.key.upper()}."
            )

    # 2. Extract all table names, excluding CTE names
    cte_names = set(cte.alias_or_name for cte in parsed.find_all(exp.CTE))
    table_names = set()

    def visit(node: exp.Expression):
        if isinstance(node, exp.Table):
            if node.name not in cte_names:
                table_names.add(node.name)
        for arg in node.args.values():
            if isinstance(arg, exp.Expression):
                visit(arg)
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, exp.Expression):
                        visit(item)

    visit(parsed)

    # 3. Validate extracted tables against allowed tables
    disallowed_tables = table_names - allowed_tables
    if disallowed_tables:
        raise ValueError(f"Disallowed tables found: {disallowed_tables}")

    return table_names

