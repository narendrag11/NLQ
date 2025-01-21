from uuid import UUID, uuid4
from sqlmodel import Session, select
import random
from models import *
from db import engine
from sqlalchemy import desc
import sys

def generate_random_vector_with_dim(dim):
    return [random.random() for _ in range(dim)]


# Get all the domain IDs in the table currently
with Session(engine) as session:
    domain_ids = session.exec(select(Domain.id)).all()
    print("Fetched domain_ids:", domain_ids)

if not domain_ids:
    print("No domain IDs found. Please populate the Domain table.")
    sys.exit(1)

for i in range(100):
    domain_embedding = DomainEmbedding(domain_id=random.choice(domain_ids), embedding=generate_random_vector_with_dim(768))
    with Session(engine) as session:
        session.add(domain_embedding)
        session.commit()

# Get all the table IDs in the table currently
with Session(engine) as session:
    table_ids = session.exec(select(Datasource.id)).all()
    print("Fetched table_ids:", table_ids)

if not table_ids:
    print("No table IDs found. Please populate the Datasource table.")
    sys.exit(1)

for i in range(100):
    table_embedding = TableEmbedding(table_id=random.choice(table_ids), embedding=generate_random_vector_with_dim(768))
    with Session(engine) as session:
        session.add(table_embedding)
        session.commit()
