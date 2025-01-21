from sqlalchemy import select,create_engine
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from models import *
# from datasource_agent import filtered_datasources_id
# from domain_agent import intent_embedding

def column_agent(datasource_ids: List[UUID], embedding_vector, engine, limit=5):
   
    with Session(engine) as session:
        # Step 1: Filter Column IDs associated with the given datasource IDs
        column_ids_query = (
            select(DataColumn.id, DataColumn.column_dec)
            .where(DataColumn.datasource_id.in_(datasource_ids))
        )
        column_results = session.execute(column_ids_query).all()
        column_ids = [row[0] for row in column_results]

        # Step 2: Perform similarity search on ColumnEmbedding
        if column_ids:
            stmt = (
                select(
                    ColumnEmbedding.column_id,
                    DataColumn.column_dec.label("column_description"),
                    (1 - ColumnEmbedding.embedding.cosine_distance(embedding_vector)).label("similarity_score")
                )
                .join(DataColumn, ColumnEmbedding.column_id == DataColumn.id)
                .where(ColumnEmbedding.column_id.in_(column_ids))
                .order_by((1 - ColumnEmbedding.embedding.cosine_distance(embedding_vector)).desc())
                .limit(limit)
            )

            results = session.execute(stmt).all()

            # Step 3: Return results
            return [(row.column_id, row.column_description, row.similarity_score) for row in results]
        else:
            print("No columns found for the given datasource IDs.")
            return []


# DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"
# engine = create_engine(DATABASE_URL)



# # Call the column_agent function
# results = column_agent(filtered_datasources_id, intent_embedding, engine, limit=5)

# # Print the results
# for column_id, column_description, similarity_score in results:
#     print(f"Column ID: {column_id}, Description: {column_description}, Similarity Score: {similarity_score:.4f}")
