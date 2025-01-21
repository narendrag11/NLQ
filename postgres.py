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
    # allowed_tables: set, 
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
    # disallowed_tables = table_names - allowed_tables
    # if disallowed_tables:
    #     raise ValueError(f"Disallowed tables found: {disallowed_tables}")

    return table_names



if __name__ == "__main__":
   
#     bigquery_working_queries = [
#     # Basic SELECT statements
#     "SELECT * FROM `your-project.your_dataset.your_table` LIMIT 10;",
#     "SELECT column1, column2 FROM `your-project.your_dataset.your_table`;",
#     "SELECT DISTINCT column1 FROM `your-project.your_dataset.your_table`;",

#     # WHERE clause and operators
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE numeric_column > 100;",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE string_column = 'value' AND date_column > DATE('2023-01-01');",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE string_column IN ('value1', 'value2');",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE string_column LIKE 'prefix%';",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE numeric_column BETWEEN 50 AND 150;",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE nullable_column IS NULL;",

#     # Aggregate functions
#     "SELECT COUNT(*) FROM `your-project.your_dataset.your_table`;",
#     "SELECT AVG(numeric_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT SUM(numeric_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT MAX(numeric_column), MIN(numeric_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT COUNT(DISTINCT string_column) FROM `your-project.your_dataset.your_table`;",

#     # GROUP BY and HAVING
#     "SELECT string_column, AVG(numeric_column) FROM `your-project.your_dataset.your_table` GROUP BY string_column;",
#     "SELECT string_column, COUNT(*) FROM `your-project.your_dataset.your_table` GROUP BY string_column HAVING COUNT(*) > 5;",

#     # ORDER BY and LIMIT/OFFSET
#     "SELECT * FROM `your-project.your_dataset.your_table` ORDER BY column1 ASC, column2 DESC LIMIT 20 OFFSET 10;",

#     # Joins
#     "SELECT t1.column1, t2.column2 FROM `your-project.your_dataset.table1` AS t1 JOIN `your-project.your_dataset.table2` AS t2 ON t1.join_column = t2.join_column;",
#     "SELECT t1.column1, t2.column2 FROM `your-project.your_dataset.table1` AS t1 LEFT JOIN `your-project.your_dataset.table2` AS t2 ON t1.join_column = t2.join_column;",
#     "SELECT t1.column1, t2.column2 FROM `your-project.your_dataset.table1` AS t1 RIGHT JOIN `your-project.your_dataset.table2` AS t2 ON t1.join_column = t2.join_column;",
#     "SELECT t1.column1, t2.column2 FROM `your-project.your_dataset.table1` AS t1 FULL OUTER JOIN `your-project.your_dataset.table2` AS t2 ON t1.join_column = t2.join_column;",
#     "SELECT e1.first_name, e2.first_name AS manager_name FROM `your-project.your_dataset.employees` e1 JOIN `your-project.your_dataset.employees` e2 ON e1.manager_id = e2.id;", #Self Join

#     # Subqueries
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE numeric_column > (SELECT AVG(numeric_column) FROM `your-project.your_dataset.your_table`);",
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE string_column IN (SELECT string_column FROM `your-project.your_dataset.another_table`);",
#     "SELECT d.department, (SELECT COUNT(*) FROM `your-project.your_dataset.employees` WHERE e.department_id=employees.department_id) FROM `your-project.your_dataset.departments` e;", #Correlated subquery

#     # Common Table Expressions (CTEs)
#     "WITH AvgSalaries AS (SELECT department, AVG(salary) AS avg_salary FROM `your-project.your_dataset.employees` GROUP BY department) SELECT * FROM AvgSalaries WHERE avg_salary > 60000;",

#     # Window functions
#     "SELECT column1, numeric_column, RANK() OVER (ORDER BY numeric_column DESC) AS rank FROM `your-project.your_dataset.your_table`;",
#     "SELECT column1, numeric_column, DENSE_RANK() OVER (PARTITION BY string_column ORDER BY numeric_column DESC) AS dense_rank FROM `your-project.your_dataset.your_table`;",
#     "SELECT column1, numeric_column, ROW_NUMBER() OVER (ORDER BY date_column) AS row_num FROM `your-project.your_dataset.your_table`;",
#     "SELECT column1, numeric_column, LAG(numeric_column, 1, 0) OVER (ORDER BY date_column) AS prev_value FROM `your-project.your_dataset.your_table`;",
#     "SELECT column1, numeric_column, AVG(numeric_column) OVER (PARTITION BY string_column) AS avg_value FROM `your-project.your_dataset.your_table`;",

#     # String functions
#     "SELECT UPPER(string_column), LOWER(string_column), LENGTH(string_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT SUBSTR(string_column, 1, 5) FROM `your-project.your_dataset.your_table`;",
#     "SELECT CONCAT(string_column, ' suffix') FROM `your-project.your_dataset.your_table`;",
#     "SELECT TRIM(string_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT REGEXP_EXTRACT(string_column, r'pattern') FROM `your-project.your_dataset.your_table`;",
#     "SELECT REGEXP_CONTAINS(string_column, r'pattern') FROM `your-project.your_dataset.your_table`;",

#     # Date/Time functions
#     "SELECT CURRENT_DATE(), CURRENT_TIMESTAMP();",
#     "SELECT EXTRACT(YEAR FROM date_column), EXTRACT(MONTH FROM date_column) FROM `your-project.your_dataset.your_table`;",
#     "SELECT DATE_ADD(date_column, INTERVAL 1 DAY), DATE_SUB(date_column, INTERVAL 1 WEEK) FROM `your-project.your_dataset.your_table`;",
#     "SELECT DATE_DIFF(date_column, DATE('2023-01-01'), DAY) FROM `your-project.your_dataset.your_table`;",

#     # Conditional expressions
#     "SELECT column1, CASE WHEN numeric_column > 100 THEN 'High' WHEN numeric_column > 50 THEN 'Medium' ELSE 'Low' END AS value_level FROM `your-project.your_dataset.your_table`;",
#     "SELECT COALESCE(nullable_column, 'default_value') FROM `your-project.your_dataset.your_table`;",

#     # Array functions
#     "SELECT ARRAY[1, 2, 3];",
#     "SELECT ARRAY_CONCAT([1, 2], [3, 4]);",
#     "SELECT UNNEST([1, 2, 3]);",
#     "SELECT column1, ARRAY_AGG(column2) AS aggregated_column FROM `your-project.your_dataset.your_table` GROUP BY column1", #Array aggregation

#     # JSON functions
#     "SELECT TO_JSON_STRING(STRUCT(1 AS a, 'text' AS b));",

#     #Geographic functions
#     "SELECT ST_AsText(ST_GeogFromText('POINT(0 0)'));",
#     "SELECT ST_Contains(ST_GeogFromText('POLYGON((-122 37, -123 37, -123 38, -122 38, -122 37))'), ST_GeogPoint(-122.5, 37.5));"

# ]

#     bigquery_non_working_queries = [
#     # Postgres/Standard SQL syntax that doesn't work directly in BigQuery
#     "SELECT * FROM your_table;",  # Missing backticks and project/dataset
#     "SELECT * FROM your_project.your_dataset.your_table WHERE string_column ILIKE 'prefix%';", # ILIKE for case-insensitive LIKE
#     "SELECT SUBSTRING(string_column FROM 1 FOR 5) FROM `your-project.your_dataset.your_table`;", # SUBSTRING with FROM/FOR
#     "SELECT string_column || ' suffix' FROM `your-project.your_dataset.your_table`;", # String concatenation with ||
#     "SELECT CURRENT_DATE, CURRENT_TIME, NOW();", # Multiple date/time functions in one SELECT
#     "SELECT EXTRACT(MONTH FROM hire_date);", #No need for parenthesis in extract
#     "SELECT hire_date + INTERVAL '1 year';", # Interval arithmetic syntax
#     "SELECT AGE(hire_date);",  # AGE function
#     "SELECT COALESCE(manager_id, 0) FROM `your-project.your_dataset.employees`;", #COALESCE with numeric default needs explicit casting if the column is of type NUMERIC
#     "SELECT ARRAY_APPEND(ARRAY[1,2],3);",#Append array
#     "SELECT '{\"a\":1, \"b\":2}'::jsonb ->> 'a';", #Extract json value
#     "CREATE EXTENSION IF NOT EXISTS postgis;", #Enable postgis extension
#     "SELECT ST_AsText(ST_GeomFromText('POINT(0 0)', 4326));", # SRID specification within ST_GeomFromText
#     "SELECT * FROM your_table PARTITION BY column1", # Partitioning syntax differs significantly
#     "SELECT column1 FROM your_table WHERE column1 LIKE '%[0-9]%'", # Regex syntax differs, use REGEXP_CONTAINS in BigQuery, ~ in Postgres
#     "SELECT GENERATE_UUID()", # UUID generation syntax is different
#     "SELECT DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)", # Date/time manipulation syntax differs
#     "CREATE OR REPLACE TABLE your_table AS SELECT * FROM another_table", # CREATE OR REPLACE syntax differs
#     "SELECT * FROM your_table FOR SYSTEM_TIME AS OF TIMESTAMP '2023-10-26 10:00:00'" # Time travel syntax differs
#     "SELECT * FROM employees UNPIVOT (value FOR attribute IN (col1, col2, col3));", #UNPIVOT

#     # Syntax errors or invalid constructs
#     "SELECT * FROM `your-project.your_dataset.your_table` WHER numeric_column > 100;", # Typo in WHERE
#     "SELECT FROM `your-project.your_dataset.your_table`;", # Missing SELECT list
#     "GROUP BY string_column HAVING COUNT(*) > 5;", # Missing SELECT
#     "SELECT column1, numeric_column RANK() OVER (ORDER BY numeric_column DESC) AS rank FROM `your-project.your_dataset.your_table`;", # Missing comma before RANK()
#     "SELECT * FROM your_project.your_dataset.table1 JOIN your_project.your_dataset.table2 ON join_column;", # Missing join condition
#     "SELECT column1, numeric_column, LAG(numeric_column, 1, 0) OVER ORDER BY date_column AS prev_value FROM `your-project.your_dataset.your_table`;", # Missing parentheses around ORDER BY in window function
#     "SELECT JSON_EXTRACT('{\"a\": 1}', 'a');", #Incorrect usage of json extract
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column ~* 'pattern';", #Case-insensitive regex operator
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column SIMILAR TO 'pattern';", #SIMILAR TO operator
#     "SELECT * FROM `your-project.your_dataset.your_table` EXCEPT SELECT * FROM `your-project.your_dataset.another_table`;", #EXCEPT operator is replaced by EXCEPT DISTINCT
#     "SELECT * FROM `your-project.your_dataset.your_table` INTERSECT SELECT * FROM `your-project.your_dataset.another_table`;", #INTERSECT operator is replaced by INTERSECT DISTINCT
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column LIKE '%[0-9]%';", #Character classes in LIKE
#     "SELECT column1 FROM `your-project.your_dataset.your_table` ORDER BY column1 LIMIT '10';", #Limit clause expects a number not string
#     "SELECT column1 FROM `your-project.your_dataset.your_table` OFFSET '10';", #Offset clause expects a number not string
#     "SELECT column1 FROM `your-project.your_dataset.your_table` FETCH FIRST 10 ROW ONLY;", #FETCH FIRST syntax
#     "SELECT column1 FROM `your-project.your_dataset.your_table` FOR UPDATE;", #FOR UPDATE lock
#     "BEGIN TRANSACTION;", #Transaction block
#     "COMMIT;", #Commit transaction
#     "ROLLBACK;", #Rollback transaction
#     "SAVEPOINT my_savepoint;", #Savepoint
#     "RELEASE SAVEPOINT my_savepoint;", #Release savepoint
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column ~~ 'pattern';", #Case-sensitive regex operator
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column !~ 'pattern';", #Case-sensitive regex not match operator
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column !~* 'pattern';", #Case-insensitive regex not match operator
#     "SELECT * FROM `your-project.your_dataset.your_table` WHERE some_column NOT SIMILAR TO 'pattern';", #NOT SIMILAR TO operator
# ]

# # print("\nBigQuery Non-Working Queries:\n", "\n".join(bigquery_non_working_queries))

#     all_queries=bigquery_working_queries+bigquery_non_working_queries
#     allowed_tables=["employees","your_table",'table1', 'table2','another_table',"departments"]
    sql_queries = [
    # CREATE statements
    "CREATE TABLE employees (id SERIAL PRIMARY KEY, name VARCHAR(100), salary NUMERIC, department_id INT);",
    "CREATE INDEX idx_department ON employees(department_id);",
    "CREATE VIEW employee_salaries AS SELECT name, salary FROM employees;",
    "CREATE DATABASE company_db;",
    "CREATE SCHEMA hr;",

    # INSERT statements
    "INSERT INTO employees (name, salary, department_id) VALUES ('John Doe', 50000, 1);",
    "INSERT INTO employees (name, salary, department_id) VALUES ('Jane Doe', 60000, 2), ('Mark Smith', 70000, 3);",
    "INSERT INTO departments (id, name) SELECT id, department_name FROM department_temp;",
    "INSERT INTO logs (event_time, event_type) VALUES (CURRENT_TIMESTAMP, 'login');",

    # UPDATE statements
    "UPDATE employees SET salary = salary + 5000 WHERE department_id = 1;",
    "UPDATE employees SET name = 'John Smith' WHERE id = 101;",
    "UPDATE employees SET department_id = 3 WHERE department_id = 2;",

    # DELETE statements
    "DELETE FROM employees WHERE salary < 30000;",
    "DELETE FROM logs WHERE event_time < '2023-01-01';",

    # ALTER statements
    "ALTER TABLE employees ADD COLUMN age INT;",
    "ALTER TABLE employees DROP COLUMN age;",
    "ALTER TABLE employees ALTER COLUMN salary TYPE FLOAT;",
    "ALTER TABLE employees RENAME COLUMN salary TO annual_salary;",
    "ALTER TABLE employees RENAME TO staff;",

    # DROP statements
    "DROP TABLE employees;",
    "DROP VIEW employee_salaries;",
    "DROP INDEX idx_department;",
    "DROP SCHEMA hr CASCADE;",
    "DROP DATABASE company_db;",

    # TRUNCATE statements
    "TRUNCATE TABLE employees;",
    "TRUNCATE TABLE logs RESTART IDENTITY;",

    # MERGE (UPSERT) statements
    "MERGE INTO employees e USING new_employees ne ON e.id = ne.id WHEN MATCHED THEN UPDATE SET salary = ne.salary WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (ne.id, ne.name, ne.salary);",

    # GRANT and REVOKE
    "GRANT SELECT, INSERT ON employees TO user_role;",
    "REVOKE DELETE ON employees FROM user_role;",
    "GRANT ALL PRIVILEGES ON DATABASE company_db TO admin_user;",
    "REVOKE CONNECT ON DATABASE company_db FROM public;",

    # ANALYZE and VACUUM
    "ANALYZE employees;",
    "VACUUM FULL employees;",

    # SET operations
    "SET search_path TO hr, public;",
    "SET statement_timeout = 3000;",

    # Miscellaneous
    "BEGIN TRANSACTION;",
    "COMMIT;",
    "ROLLBACK;",
    "LOCK TABLE employees IN ACCESS EXCLUSIVE MODE;",
    "DISCARD ALL;",
    "REASSIGN OWNED BY old_user TO new_user;",
    "COMMENT ON TABLE employees IS 'Table containing employee data';"
]

    for sql_query in sql_queries:
            print("=======================")
            print()

            print(sql_query)
            table_names = extract_and_validate_tables(sql_query,dialect="bigquery")
            if len(table_names)==0:
                print(f"Extracted Table Names: {table_names}")
                print()

                print("*************************")
