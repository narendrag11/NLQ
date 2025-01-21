# add_data.py

from sqlmodel import Session, select
from db import engine, create_db_and_tables
from models import (
    Workspace,
    Domain,
    Datasource,
    DomainDatasource,
    DataColumn,
)

def add_dummy_data():
    # Create (or ensure) the DB tables exist
    create_db_and_tables()

    # Open a session
    with Session(engine) as session:
        # 1) Create a Workspace
        workspace = Workspace(name="Workspace A")
        session.add(workspace)
        session.commit()      # Commit to get the workspace ID
        session.refresh(workspace)

        # 2) Create a Domain (linked to Workspace)
        domain = Domain(
            name="Employee Benefits",
            description="Holds information about employee benefit plans",
            purpose="Used for tracking and analyzing benefits",
            workspace_id=workspace.id,
        )
        session.add(domain)
        session.commit()
        session.refresh(domain)

        # 3) Create a Datasource
        datasource = Datasource(description="HR Database")
        session.add(datasource)
        session.commit()
        session.refresh(datasource)

        # 4) Associate the Domain with the Datasource (DomainDatasource)
        domain_datasource = DomainDatasource(
            domain_id=domain.id,
            datasource_id=datasource.id
        )
        session.add(domain_datasource)
        session.commit()
        session.refresh(domain_datasource)

        # 5) Create a DataColumn (linked to the same Datasource)
        data_column = DataColumn(
            column_dec="Employee Salary Details",
            datasource_id=datasource.id
        )
        session.add(data_column)
        session.commit()
        session.refresh(data_column)

        print("Dummy data inserted successfully!")
        print(f"Workspace: {workspace}")
        print(f"Domain: {domain}")
        print(f"Datasource: {datasource}")
        print(f"DomainDatasource: {domain_datasource}")
        print(f"DataColumn: {data_column}")


if __name__ == "__main__":
    add_dummy_data()
