import sqlglot
from sqlglot.expressions import Table, Column, Select, Union, Exists, CTE, Delete, Where, Binary

def parse_query(query: str):
    parsed = sqlglot.parse_one(query)
    table_info = {}
    cte_definitions = set()  # Track CTE names to exclude them as tables

    def add_column_to_table(table_name, column_name):
        if table_name not in table_info:
            table_info[table_name] = set()
        table_info[table_name].add(column_name)

    def resolve_table_name(col_table, scope):
        """Resolve the actual table name from a column's table reference"""
        for table_expr in scope.find_all(Table):
            if table_expr.alias == col_table:
                return table_expr.name
            elif table_expr.name == col_table:
                return table_expr.name
        return col_table

    def process_where_conditions(where_expr, scope):
        """Process columns in WHERE conditions"""
        if isinstance(where_expr, Binary):
            for col in where_expr.find_all(Column):
                if col.table:
                    table_name = resolve_table_name(col.table, scope)
                    if table_name not in cte_definitions:
                        add_column_to_table(table_name, col.name)

    def process_select(select_expr, is_subquery=False):
        # Register tables first
        for table_expr in select_expr.find_all(Table):
            if table_expr.name not in cte_definitions:
                table_info[table_expr.name] = set()

        # Process all columns in the SELECT statement
        for col_expr in select_expr.find_all(Column):
            if col_expr.table:
                table_name = resolve_table_name(col_expr.table, select_expr)
                if table_name not in cte_definitions:
                    add_column_to_table(table_name, col_expr.name)
            else:
                # Unqualified column - add to all tables in current scope
                tables_in_scope = [t for t in select_expr.find_all(Table) 
                                 if t.name not in cte_definitions]
                for table_expr in tables_in_scope:
                    add_column_to_table(table_expr.name, col_expr.name)

        # Process WHERE conditions
        where_clause = select_expr.find(Where)
        if where_clause:
            process_where_conditions(where_clause.this, select_expr)

    def traverse_query(expr):
        # Handle CTEs first
        for cte_expr in expr.find_all(CTE):
            cte_definitions.add(cte_expr.alias)

        # Handle different query types
        if isinstance(expr, Select):
            process_select(expr)
        elif isinstance(expr, Delete):
            # For DELETE statements, process the table and WHERE clause
            table_name = expr.find(Table).name
            where_clause = expr.find(Where)
            if where_clause:
                process_where_conditions(where_clause.this, expr)
        elif isinstance(expr, Union):
            process_select(expr.this)
            process_select(expr.expression)

        # Process subqueries in WHERE EXISTS
        for exists_expr in expr.find_all(Exists):
            subquery = exists_expr.this
            if isinstance(subquery, Select):
                process_select(subquery, is_subquery=True)

        # Traverse child expressions
        for child in expr.expressions:
            traverse_query(child)

    # Start processing
    traverse_query(parsed)

    # Remove CTEs from final output and convert sets to sorted lists
    final_info = {table: sorted(columns) 
                 for table, columns in table_info.items() 
                 if table not in cte_definitions}
    
    return final_info

# Example usage:
if __name__ == "__main__":
    # Test with a complex query
    query = """
    SELECT 
        t1.column1, 
        t2.column2,
        t1.column3
    FROM table1 t1 
    JOIN table2 t2 ON t1.id = t2.id
    WHERE t1.column1 > 10;
    """
    test_queries = [
        # 1. Basic SELECT (unqualified columns)
        "SELECT column1, column2, column3 FROM table_name;",

        # 2. SELECT with WHERE condition (unqualified columns)
        "SELECT column1, column2 FROM table_name WHERE column3 = 'value';",

        # 3. ORDER BY clause (unqualified columns)
        "SELECT column1, column2, column3 FROM table_name ORDER BY column1 ASC, column2 DESC;",

        # 4. DISTINCT
        "SELECT DISTINCT column1, column2 FROM table_name;",

        # 5. GROUP BY with HAVING
        "SELECT column1, COUNT(column2) FROM table_name GROUP BY column1 HAVING COUNT(column2) > 1;",

        # 6. LIMIT
        "SELECT column1, column2 FROM table_name LIMIT 10;",

        # 7. INNER JOIN (qualified columns)
        "SELECT a.column1, b.column2 FROM table1 a INNER JOIN table2 b ON a.common_column = b.common_column;",

        # 8. LEFT JOIN
        "SELECT a.column1, b.column2 FROM table1 a LEFT JOIN table2 b ON a.common_column = b.common_column;",

        # 9. RIGHT JOIN
        "SELECT a.column1, b.column2 FROM table1 a RIGHT JOIN table2 b ON a.common_column = b.common_column;",

        # 10. FULL OUTER JOIN
        "SELECT a.column1, b.column2 FROM table1 a FULL OUTER JOIN table2 b ON a.common_column = b.common_column;",

        # 11. UNION of two SELECTs (each branch has one table so unqualified columns are assigned)
        "SELECT column1, column2 FROM table1 UNION SELECT column1, column2 FROM table2;",

        # 12. Subquery in WHERE clause (unqualified columns)
        "SELECT column1, column2 FROM table_name WHERE column3 = (SELECT MAX(column3) FROM table_name);",

        # 13. CASE WHEN expression (unqualified columns)
        "SELECT column1, CASE WHEN column2 > 10 THEN 'High' ELSE 'Low' END AS category FROM table_name;",

        # 14. IN clause
        "SELECT column1, column2 FROM table_name WHERE column3 IN ('value1', 'value2', 'value3');",

        # 15. BETWEEN clause
        "SELECT column1, column2 FROM table_name WHERE column3 BETWEEN '2023-01-01' AND '2023-12-31';",

        # 16. LIKE clause
        "SELECT column1, column2 FROM table_name WHERE column3 LIKE 'prefix%';",

        # 17. Aggregation (with unqualified columns)
        "SELECT SUM(column1) AS total, AVG(column2) AS average FROM table_name;",

        # 18. Window function (unqualified columns)
        "SELECT column1, column2, ROW_NUMBER() OVER (PARTITION BY column1 ORDER BY column2) AS row_num FROM table_name;",

        # 19. CROSS JOIN (qualified columns)
        "SELECT a.column1, b.column2 FROM table1 a CROSS JOIN table2 b;",

        # 20. EXISTS with subquery (qualified columns)
        "SELECT column1, column2 FROM table_name WHERE EXISTS (SELECT 1 FROM another_table WHERE another_table.column3 = table_name.column3);",

        # 21. Subquery in SELECT list (qualified columns)
        "SELECT column1, (SELECT COUNT(*) FROM another_table WHERE another_table.column2 = table_name.column1) AS count_column FROM table_name;",

        # 22. Simple CTE – the underlying physical table is 'orders'
        """
        WITH top_customers AS (
            SELECT customer_id, SUM(amount) AS total_spent
            FROM orders
            GROUP BY customer_id
            HAVING SUM(amount) > 1000
        )
        SELECT customer_id, total_spent FROM top_customers;
        """,

        # 23. CTE with multiple queries – both CTEs come from 'orders'
        """
        WITH top_customers AS (
            SELECT customer_id, SUM(amount) AS total_spent
            FROM orders
            GROUP BY customer_id
            HAVING SUM(amount) > 1000
        ),
        recent_orders AS (
            SELECT customer_id, order_date, amount
            FROM orders
            WHERE order_date >= '2024-01-01'
        )
        SELECT tc.customer_id, tc.total_spent, ro.order_date, ro.amount
        FROM top_customers tc
        JOIN recent_orders ro ON tc.customer_id = ro.customer_id;
        """,

        # 24. Window functions with a single physical table (unqualified columns)
        """
        SELECT 
            employee_id, 
            salary, 
            LAG(salary, 1) OVER (PARTITION BY department_id ORDER BY hire_date) AS prev_salary, 
            LEAD(salary, 1) OVER (PARTITION BY department_id ORDER BY hire_date) AS next_salary 
        FROM employees;
        """,

        # 25. Another window function example
        """
        SELECT 
            employee_id, 
            department_id, 
            salary, 
            PERCENT_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank 
        FROM employees;
        """,

        # 26. NTILE window function example
        """
        SELECT 
            employee_id, 
            salary, 
            NTILE(4) OVER (ORDER BY salary) AS quartile 
        FROM employees;
        """,

        # 27. Recursive CTE – physical tables from both the base and recursive parts.
        """
        WITH RECURSIVE employee_hierarchy AS (
            SELECT employee_id, manager_id, name, 1 AS level 
            FROM employees WHERE manager_id IS NULL
            UNION ALL
            SELECT e.employee_id, e.manager_id, e.name, eh.level + 1 
            FROM employees e
            JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
        )
        SELECT * FROM employee_hierarchy;
        """,

        # 28. DELETE query using a CTE – physical table is table_name.
        """
        WITH duplicate_cte AS (
            SELECT 
                ROW_NUMBER() OVER (PARTITION BY column1, column2 ORDER BY id) AS row_num, 
                id 
            FROM table_name
        )
        DELETE FROM table_name WHERE id IN (SELECT id FROM duplicate_cte WHERE row_num > 1);
        """,

        # 29. Pivot-like query (GROUP BY) – unqualified columns.
        """
        SELECT 
            employee_id, 
            SUM(CASE WHEN department = 'Sales' THEN salary ELSE 0 END) AS Sales_Salary,
            SUM(CASE WHEN department = 'IT' THEN salary ELSE 0 END) AS IT_Salary,
            SUM(CASE WHEN department = 'HR' THEN salary ELSE 0 END) AS HR_Salary
        FROM employees
        GROUP BY employee_id;
        """,

        # 30. Recursive CTE for Fibonacci series.
        """
        WITH RECURSIVE Fibonacci AS (
            SELECT 0 AS n, 0 AS fibonacci
            UNION ALL
            SELECT 1, 1
            UNION ALL
            SELECT n + 1, f1.fibonacci + f2.fibonacci
            FROM Fibonacci f1, Fibonacci f2
            WHERE f1.n + 1 = f2.n
            LIMIT 10
        )
        SELECT * FROM Fibonacci;
        """
    ]
    i=0
    for query in test_queries:
        print("count",i)
        i+=1
        print(query)
        result = parse_query(query)
        # print("*"*10)
        # print("Tables and their columns:")
        for table, columns in result.items():
            # print("-"*10)
            print(f"{table}: {columns}")
            # print("-"*10)


ouput=["""
count 0
SELECT column1, column2, column3 FROM table_name;
table_name: ['column1', 'column2', 'column3']
count 1
SELECT column1, column2 FROM table_name WHERE column3 = 'value';
table_name: ['column1', 'column2', 'column3']
count 2
SELECT column1, column2, column3 FROM table_name ORDER BY column1 ASC, column2 DESC;
table_name: ['column1', 'column2', 'column3']
count 3
SELECT DISTINCT column1, column2 FROM table_name;
table_name: ['column1', 'column2']
count 4
SELECT column1, COUNT(column2) FROM table_name GROUP BY column1 HAVING COUNT(column2) > 1;
table_name: ['column1', 'column2']
count 5
SELECT column1, column2 FROM table_name LIMIT 10;
table_name: ['column1', 'column2']
count 6
SELECT a.column1, b.column2 FROM table1 a INNER JOIN table2 b ON a.common_column = b.common_column;
table1: ['column1', 'common_column']
table2: ['column2', 'common_column']
count 7
SELECT a.column1, b.column2 FROM table1 a LEFT JOIN table2 b ON a.common_column = b.common_column;
table1: ['column1', 'common_column']
table2: ['column2', 'common_column']
count 8
SELECT a.column1, b.column2 FROM table1 a RIGHT JOIN table2 b ON a.common_column = b.common_column;
table1: ['column1', 'common_column']
table2: ['column2', 'common_column']
count 9
SELECT a.column1, b.column2 FROM table1 a FULL OUTER JOIN table2 b ON a.common_column = b.common_column;
table1: ['column1', 'common_column']
table2: ['column2', 'common_column']
count 10
SELECT column1, column2 FROM table1 UNION SELECT column1, column2 FROM table2;
table1: ['column1', 'column2']
table2: ['column1', 'column2']
count 11
SELECT column1, column2 FROM table_name WHERE column3 = (SELECT MAX(column3) FROM table_name);
table_name: ['column1', 'column2', 'column3']
count 12
SELECT column1, CASE WHEN column2 > 10 THEN 'High' ELSE 'Low' END AS category FROM table_name;
table_name: ['column1', 'column2']
count 13
SELECT column1, column2 FROM table_name WHERE column3 IN ('value1', 'value2', 'value3');
table_name: ['column1', 'column2', 'column3']
count 14
SELECT column1, column2 FROM table_name WHERE column3 BETWEEN '2023-01-01' AND '2023-12-31';
table_name: ['column1', 'column2', 'column3']
count 15
SELECT column1, column2 FROM table_name WHERE column3 LIKE 'prefix%';
table_name: ['column1', 'column2', 'column3']
count 16
SELECT SUM(column1) AS total, AVG(column2) AS average FROM table_name;
table_name: ['column1', 'column2']
count 17
SELECT column1, column2, ROW_NUMBER() OVER (PARTITION BY column1 ORDER BY column2) AS row_num FROM table_name;
table_name: ['column1', 'column2']
count 18
SELECT a.column1, b.column2 FROM table1 a CROSS JOIN table2 b;
table1: ['column1']
table2: ['column2']
count 19
SELECT column1, column2 FROM table_name WHERE EXISTS (SELECT 1 FROM another_table WHERE another_table.column3 = table_name.column3);
table_name: ['column1', 'column2', 'column3']
another_table: ['column3']
count 20
SELECT column1, (SELECT COUNT(*) FROM another_table WHERE another_table.column2 = table_name.column1) AS count_column FROM table_name;
table_name: ['column1']
another_table: ['column1', 'column2']
count 21

        WITH top_customers AS (
            SELECT customer_id, SUM(amount) AS total_spent
            FROM orders
            GROUP BY customer_id
            HAVING SUM(amount) > 1000
        )
        SELECT customer_id, total_spent FROM top_customers;
        
orders: ['amount', 'customer_id', 'total_spent']
count 22

        WITH top_customers AS (
            SELECT customer_id, SUM(amount) AS total_spent
            FROM orders
            GROUP BY customer_id
            HAVING SUM(amount) > 1000
        ),
        recent_orders AS (
            SELECT customer_id, order_date, amount
            FROM orders
            WHERE order_date >= '2024-01-01'
        )
        SELECT tc.customer_id, tc.total_spent, ro.order_date, ro.amount
        FROM top_customers tc
        JOIN recent_orders ro ON tc.customer_id = ro.customer_id;
        
orders: ['amount', 'customer_id', 'order_date']
count 23

        SELECT 
            employee_id, 
            salary, 
            LAG(salary, 1) OVER (PARTITION BY department_id ORDER BY hire_date) AS prev_salary, 
            LEAD(salary, 1) OVER (PARTITION BY department_id ORDER BY hire_date) AS next_salary 
        FROM employees;
        
employees: ['department_id', 'employee_id', 'hire_date', 'salary']
count 24

        SELECT 
            employee_id, 
            department_id, 
            salary, 
            PERCENT_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank 
        FROM employees;
        
employees: ['department_id', 'employee_id', 'salary']
count 25

        SELECT 
            employee_id, 
            salary, 
            NTILE(4) OVER (ORDER BY salary) AS quartile 
        FROM employees;
        
employees: ['employee_id', 'salary']
count 26

        WITH RECURSIVE employee_hierarchy AS (
            SELECT employee_id, manager_id, name, 1 AS level 
            FROM employees WHERE manager_id IS NULL
            UNION ALL
            SELECT e.employee_id, e.manager_id, e.name, eh.level + 1 
            FROM employees e
            JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
        )
        SELECT * FROM employee_hierarchy;
        
employees: ['employee_id', 'manager_id', 'name']
count 27

        WITH duplicate_cte AS (
            SELECT 
                ROW_NUMBER() OVER (PARTITION BY column1, column2 ORDER BY id) AS row_num, 
                id 
            FROM table_name
        )
        DELETE FROM table_name WHERE id IN (SELECT id FROM duplicate_cte WHERE row_num > 1);
        
count 28

        SELECT 
            employee_id, 
            SUM(CASE WHEN department = 'Sales' THEN salary ELSE 0 END) AS Sales_Salary,
            SUM(CASE WHEN department = 'IT' THEN salary ELSE 0 END) AS IT_Salary,
            SUM(CASE WHEN department = 'HR' THEN salary ELSE 0 END) AS HR_Salary
        FROM employees
        GROUP BY employee_id;
        
employees: ['department', 'employee_id', 'salary']
count 29

        WITH RECURSIVE Fibonacci AS (
            SELECT 0 AS n, 0 AS fibonacci
            UNION ALL
            SELECT 1, 1
            UNION ALL
            SELECT n + 1, f1.fibonacci + f2.fibonacci
            FROM Fibonacci f1, Fibonacci f2
            WHERE f1.n + 1 = f2.n
            LIMIT 10
        )
        SELECT * FROM Fibonacci;

"""]
