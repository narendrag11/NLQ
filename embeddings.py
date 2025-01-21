# from sqlmodel import Session, select
# # from langchain.embeddings import GeminiEmbeddings  # Replace with actual Gemini integration
# from sqlalchemy import create_engine
# from models import Domain, Datasource, DataColumn, DomainEmbedding, TableEmbedding, ColumnEmbedding
# from db import engine  \                                                                                                                                                                                                                                                                                                                                                                                                                                                
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# import os
# import getpass

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Replace with your API key
# def generate_embeddings():
#     with Session(engine) as session:
#         # 1. Embed Domain data
#         domains = session.exec(select(Domain)).all()
#         for domain in domains:
#             # -- Domain Description --
#             if domain.description:  # Only embed if description is not None/empty
#                 description_text = domain.description.strip()
#                 description_vector = embeddings.embed_query(description_text)
#                 domain_description_embedding = DomainEmbedding(
#                     domain_id=domain.id,
#                     embedding=description_vector,
#                     # text=description_text,
#                     embedding_type="description",
#                 )
#                 session.add(domain_description_embedding)

#             # -- Domain Purpose --
#             if domain.purpose:  # Only embed if purpose is not None/empty
#                 purpose_text = domain.purpose.strip()
#                 purpose_vector = embeddings.embed_query(purpose_text)
#                 domain_purpose_embedding = DomainEmbedding(
#                     domain_id=domain.id,
#                     embedding=purpose_vector,
#                     # text=purpose_text,
#                     embedding_type="purpose",
#                 )
#                 session.add(domain_purpose_embedding)

#         session.commit()
#         print(f"Embedded {len(domains)} domains successfully.")

#         # 2. Embed Datasource data
#         datasources = session.exec(select(Datasource)).all()
#         for datasource in datasources:
#             # Use datasource description as text
#             text_to_embed = (datasource.description or "").strip()
#             embedding_vector = embeddings.embed_query(text_to_embed)
#             table_embedding = TableEmbedding(
#                 table_id=datasource.id,
#                 embedding=embedding_vector,
#                 # text=text_to_embed,
#             )
#             session.add(table_embedding)

#         session.commit()
#         print(f"Embedded {len(datasources)} datasources successfully.")

#         # 3. Embed DataColumn data
#         data_columns = session.exec(select(DataColumn)).all()
#         for column in data_columns:
#             # Use column description as text
#             text_to_embed = (column.column_dec or "").strip()
#             embedding_vector = embeddings.embed_query(text_to_embed)
#             column_embedding = ColumnEmbedding(
#                 column_id=column.id,
#                 embedding=embedding_vector,
#                 # text=text_to_embed,
#             )
#             session.add(column_embedding)

#         session.commit()
#         print(f"Embedded {len(data_columns)} columns successfully.")


# if __name__ == "__main__":
#     generate_embeddings()


# # AIzaSyCSnW71acAMxW29uPX6N2EeWjnej2BG2TI