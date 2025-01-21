from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from models import *
from uuid import UUID
# from domain_agent import intent_embedding
def datasources_agent(domain_ids, query_embedding, engine, limit=5):
    with Session(engine) as session:
        
        datasource_ids_query = (
            select(DomainDatasource.datasource_id)
            .where(DomainDatasource.domain_id.in_(domain_ids))
        )
        datasource_ids = [row[0] for row in session.execute(datasource_ids_query).all()]

        if datasource_ids:
            stmt = (
                select(
                    TableEmbedding.table_id.label("datasource_embedding_id"),
                    Datasource.description.label("datasource_name"),
                    (1 - TableEmbedding.embedding.cosine_distance(query_embedding)).label("similarity_score"),
                    Datasource.id.label("datasource_id")
                )
                .join(Datasource, TableEmbedding.table_id == Datasource.id)
                .where(TableEmbedding.table_id.in_(datasource_ids))
                .order_by((1 - TableEmbedding.embedding.cosine_distance(query_embedding)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()

            
            if results:
                print("Available keys in results:", results[0]._fields)

            
            return [str(row.datasource_id) for row in results]
         
        else:
            print("No datasources found for the given domain IDs.")
            return []



# query_embedding=intent_embedding

# DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"
# engine = create_engine(DATABASE_URL)
# domains=[UUID('5acff12a-480e-4a39-824e-e87e632efc81'), UUID('a0e9f9e4-5fb4-487e-a035-80746c47e0ec'), UUID('5acff12a-480e-4a39-824e-e87e632efc81'), UUID('a0e9f9e4-5fb4-487e-a035-80746c47e0ec'), UUID('f3a9bcb0-a6ba-4a68-ba28-32b3f8f4a8d2')]

# # Convert UUIDs to strings (if required by the database schema)
# domain_ids_str = [str(uuid_obj) for uuid_obj in domains]

# # # Call the function
# results = datasources_agent(domain_ids_str, query_embedding, engine, limit=5)


# Print the results

# for datasource_id, datasource_name, similarity_score in results:
#     print(f"Datasource ID: {datasource_id}, Datasource Name: {datasource_name}, Similarity Score: {similarity_score:.4f}")


# filtered_datasources_id=[str(datasource_id) for datasource_id, datasource_name, similarity_score in results]
# print("filter_Datasourcs",filtered_datasources_id)
