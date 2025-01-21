from sqlmodel import SQLModel, create_engine
from sqlalchemy import text
from models import *

DATABASE_URL = "postgresql://postgres:password@localhost:5432/Autobi_DB"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    # Ensure the embedding schema exists
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS embedding"))
        conn.commit()
    
    # Create tables
    SQLModel.metadata.create_all(engine)
    print("Database and tables created successfully!")

create_db_and_tables()
