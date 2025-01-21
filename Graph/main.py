from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("postgresql://postgres:password@localhost:5432/Autobi_DB")
print(db.dialect)
print(db.get_usable_table_names())
result=db.run("SELECT * FROM datasource LIMIT 10;")
print(result)