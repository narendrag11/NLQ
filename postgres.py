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
        records = cursor.fetchall()
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
