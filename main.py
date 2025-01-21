from column_Agent import column_agent
from datasource_agent import datasources_agent
from domain_agent import domain_agent
from intent_agents import intent_agent
import json
from sqlalchemy import create_engine


def complete_flow(user_input,engine):
    extracted_data,embedded_vector = intent_agent(user_input)
    domain_ids=domain_agent(embedded_vector,engine,limit=2)
    datasources_ids=datasources_agent(domain_ids,embedded_vector,engine,limit=3)
    column_ids=column_agent(datasources_ids,embedded_vector,engine)
    return datasources_ids,column_ids
    

user_input="What are my sales last month?"
DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"
engine = create_engine(DATABASE_URL)
datasource,column=complete_flow(user_input,engine)
for d in datasource:
    print("datasource",datasource)


print("============++++++===============")
for c in column:
    print("column",c)